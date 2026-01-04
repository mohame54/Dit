import torch
from typing import Union, List
from .solvers import ODSolversMixin

LABEL_DICT = {
    "bird":0,
    "cat":1,
    "dog":2,
}


def sample_t_uniform(batch_sz:int):
    return torch.rand(batch_sz)


def sample_t_logit_normal(batch_sz, mu=-1.0, sigma=1.0):
   # sample s ~ Normal(mu, sigma) then t = sigmoid(s)
    s = torch.randn(batch_sz) * float(sigma) + float(mu)
    t = torch.sigmoid(s)
    return t
    

def sample_t_logit_normal_quantized(batch_sz, mu=-1.0, sigma=1.0, n_steps=1000):
    t = sample_t_logit_normal(batch_sz, mu, sigma)
    scaled = (t * n_steps).clamp(min=0.0, max=float(n_steps) - 1e-9)
    k = torch.floor(scaled).to(torch.long)
    u = torch.rand_like(t)
    t_q = (k.to(t.dtype) + u) / float(n_steps)
    return t_q


class RFDiffusion(ODSolversMixin):
    def __init__(
        self,
        model,
        vae,
        n_steps=1000,
        sigma=1.0,
        mu=0.0,
        sampler_dist="logit_normal",
        sampling_method="rk"
    ):
        super(RFDiffusion, self).__init__()
        self.model = model
        self.vae = vae
        self.sigma = sigma
        self.sampler_dist = sampler_dist
        self.mu = mu
        self.n_steps = n_steps
        self.set_sampling_method(sampling_method)

    def set_sigma(self, sigma: float):
        self.sigma = sigma

    def set_sampling_method(self, method: str):
        method = method.lower()
        if method not in ["euler", "rk"]:
            raise ValueError(f"Unknown sampling_method: {method}")
        self.sampling_method = method

    def sample_t(self, batch_sz: int, min_val=1e-6, max_val=1.0-1e-6, device="cpu"):
        if self.sampler_dist == "uniform":
            t= sample_t_uniform(batch_sz)
        elif self.sampler_dist == "logit_normal":
            t = sample_t_logit_normal_quantized(batch_sz, mu=self.mu, sigma=self.sigma, n_steps=self.n_steps)
        else:
            raise ValueError(f"Unknown sampler_dist: {self.sampler_dist}")
        t = t.clamp(min=min_val, max=max_val)
        return t.to(device)
        
    def set_sampler_dist(self, sampler_dist: str):
        self.sampler_dist = sampler_dist

    def set_model(self, model):
        self.model = model

    def to_t_inds(self, t: torch.Tensor):
        t_inds = (t * float(self.n_steps)).to(torch.int32)
        return t_inds.to(t.device)
     
    def rectified_flow_loss(
        self,
        x0,
        cond=None,
        loss_type="mse_loss",
        min_val=1e-6,
        max_val=1.0-1e-6
    ):
        """
        x0: clean images in [-1,1], shape (B,C,H,W)
        returns: scalar loss
        """
        B = x0.shape[0]
        z = torch.randn_like(x0) * self.sigma
        t = self.sample_t(B, min_val=min_val, max_val=max_val, device=x0.device)  # (B,)
        t_b = t.view(B, 1, 1, 1)
        x_t = (1.0 - t_b) * x0 + t_b * z  # linear path
        v_pred = self.model(x_t, self.to_t_inds(t), cond=cond)
        target_v = (z - x0)
        loss_fn = getattr(torch.nn.functional, loss_type)
        if loss_fn == None:
            raise ValueError(f"Unknown loss type: {loss_type}")
        loss = loss_fn(v_pred, target_v) 
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        cond=None,
        steps: int = 10,
        z_init: torch.Tensor = None,
        shape=(),
        return_traj=False,
        device="cpu",
        cfg_fac=2.0
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        if z_init is None and not len(shape):
            raise ValueError("Either z_init or shape must be provided")
        
        if z_init is None:  
            z_init = torch.randn(shape, device=device)
        shape = z_init.shape
        dt = 1.0 / float(steps)
        steps += 1 # include t=0
        x = z_init
        self.model.eval()
        t_vals = torch.linspace(self.n_steps, 0.0, steps, device=device) / self.n_steps
        traj = []
        if return_traj:
            traj.append(x.cpu())
        
        if cond is None:
            null_labels = torch.zeros((shape[0],) , dtype=torch.int32, device=device)
        else:
            null_labels = torch.zeros_like(cond)

        def model_fn(x_t, t):
            cond_tensor = cond
            x_t_inp = x_t
            t_inp = t
            if cfg_fac > 0.0 and cond is not None:
               cond_tensor = torch.cat([null_labels, cond])
               x_t_inp = torch.cat([x_t] * 2)
               t_inp = torch.cat([t]* 2)
               
            out = self.model(x_t_inp, t_inp, cond=cond_tensor)

            if cfg_fac > 0.0 and cond is not None:
               out_uncond, out_cond = out.chunk(2)
               out = out_uncond + cfg_fac * (out_cond - out_uncond)
            return out
           
        for i in range(steps - 1):  # Stop one step before to avoid index error
            t = t_vals[i]
            t_batch = torch.full((x.shape[0],), fill_value=t, device=device)
            t_batch = self.to_t_inds(t_batch)

            v = model_fn(x, t_batch)
            if self.sampling_method == "euler":
                x = self.euler_step(x, dt, v)

            elif self.sampling_method == "rk":
                t_batch_next = torch.full((x.shape[0],), fill_value=t_vals[i + 1], device=device)
                t_batch_next = self.to_t_inds(t_batch_next)
                x = self.rk2_step(x, dt, v, t_batch, t_batch_next, model_fn)
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")
            if return_traj:
                traj.append(x.cpu())
        self.model.train()
        if return_traj:
            return x, traj
        return x
    
    def generate(
        self,
        steps: int,
        labels: List[Union[int, str]],
        cfg_fac: float = 2.0,
        device: str = "cuda",
        latent_shape: List[int] = (4, 64, 64),
        label_dict: dict = LABEL_DICT,
        return_trj: bool = True     
    ):
        if len(labels) > 0 and isinstance(labels[0], str):
            labels = [label_dict[l] + 1 for l in labels]
        elif len(labels) > 0 and isinstance(labels[0], int):
            # Labels are already integers, use as-is
            labels = list(labels)
        labels_tensor = torch.tensor(labels, dtype=torch.int32, device=device)
        batch_size = labels_tensor.shape[0]
        shape = (batch_size, ) + latent_shape
        samples = self.sample(
            cond=labels_tensor,
            steps=steps,
            shape=shape,
            return_traj=return_trj,
            device=device,
            cfg_fac=cfg_fac,
        )
        if return_trj:
            latents, traj = samples
            latents = (latents * (1/ 0.18215)).to(device)
            traj = [(lat * (1/ 0.18215)).to(device) for lat in traj]
            all_latents = torch.cat([latents] + traj)
        else:
            latents = samples
            latents = latents * (1/ 0.18215)
            all_latents = latents

        
        samples = self.vae.decode(all_latents).sample
        samples = samples.cpu()
        # normalize to [0,1]
        samples = samples * 0.5 + 0.5
        samples = samples.clamp(0.0, 1.0)
        torch.cuda.empty_cache()
        if return_trj:
            return samples[:batch_size], samples[batch_size:]
        return samples
