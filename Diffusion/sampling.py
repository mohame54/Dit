import torch
from typing import Union, List
from .solvers import ODSolversMixin
from utils import unscale_latent
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
        sampling_method="rk",
        shift=1.0,
    ):
        super(RFDiffusion, self).__init__()
        self.model = model
        self.vae = vae
        self.sigma = sigma
        self.sampler_dist = sampler_dist
        self.mu = mu
        self.n_steps = n_steps
        self.shift = shift
        self.set_sampling_method(sampling_method)

    def set_model(self, model):
        self.model = model

    def set_vae(self, vae):
        self.vae = vae

    def set_sigma(self, sigma: float):
        self.sigma = float(sigma)

    def set_mu(self, mu: float):
        self.mu = float(mu)

    def set_n_steps(self, n_steps: int):
        self.n_steps = int(n_steps)

    def set_shift(self, shift: float):
        if shift <= 0:
            raise ValueError(f"shift must be > 0, got {shift}")
        self.shift = float(shift)

    def set_sampling_method(self, method: str):
        method = method.lower()
        if method not in ["euler", "rk", "rk4", "ab2"]:
            raise ValueError(
                f"Unknown sampling_method: {method!r}. "
                "Choose from 'euler', 'rk' (Heun), 'rk4', or 'ab2'."
            )
        self.sampling_method = method

    def set_sampler_dist(self, sampler_dist: str):
        if sampler_dist not in ["uniform", "logit_normal"]:
            raise ValueError(
                f"Unknown sampler_dist: {sampler_dist!r}. "
                "Choose from 'uniform' or 'logit_normal'."
            )
        self.sampler_dist = sampler_dist

    @staticmethod
    def _shift_timesteps(t: torch.Tensor, shift: float) -> torch.Tensor:
        """Apply Flux/SD3-style timestep shifting.

        Maps uniform t ∈ [0,1] to a shifted schedule via the Möbius transform:
            t_shifted = shift * t / (1 + (shift - 1) * t)

        shift=1.0  →  identity (uniform spacing, current default)
        shift=3.0  →  SD3 fixed shift; steps are denser near t=1 (noisy end)
        shift>1    →  more model evaluations in the high-noise region where
                       the velocity field is less straight, improving quality
                       at the same number of steps.
        """
        if shift == 1.0:
            return t
        return shift * t / (1.0 + (shift - 1.0) * t)

    def sample_t(self, batch_sz: int, min_val=1e-6, max_val=1.0-1e-6, device="cpu"):
        if self.sampler_dist == "uniform":
            t= sample_t_uniform(batch_sz)
        elif self.sampler_dist == "logit_normal":
            t = sample_t_logit_normal_quantized(batch_sz, mu=self.mu, sigma=self.sigma, n_steps=self.n_steps)
        else:
            raise ValueError(f"Unknown sampler_dist: {self.sampler_dist}")
        t = t.clamp(min=min_val, max=max_val)
        return t.to(device)
        
    def to_t_inds(self, t: torch.Tensor):
        # Clamp to [0, n_steps-1] so inference (which starts at t=1.0) never
        # produces index n_steps, which is outside the training distribution.
        t_inds = (t * float(self.n_steps)).to(torch.int32).clamp(0, self.n_steps - 1)
        return t_inds.to(t.device)
     
    def rectified_flow_loss(
        self,
        x0,
        cond=None,
        loss_type="mse_loss",
        min_val=1e-6,
        max_val=1.0-1e-6,
    ):
        """
        x0: clean latent tensor, shape (B,C,H,W)
        returns: scalar loss
        """
        B = x0.shape[0]
        z = torch.randn_like(x0) * self.shift + self.mu
        t = self.sample_t(B, min_val=min_val, max_val=max_val, device=x0.device)  # (B,)
        t = t.to(dtype=x0.dtype)  # Match dtype for mixed precision
        t_b = t.view(B, 1, 1, 1)
        x_t = (1.0 - t_b) * x0 + t_b * z  # linear path
        v_pred = self.model(x_t, self.to_t_inds(t), cond=cond)
        target_v = (z - x0)
       
        loss_fn = getattr(torch.nn.functional, loss_type, None)
        if loss_fn is None:
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
        cfg_fac=2.0,
        mu: float = None,
        noise_scale: float = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Force eval mode so label_dropout (training-only) never fires during
        # generation, regardless of what mode the caller left the model in.
        training = self.model.training
        self.model.eval()

        if z_init is None and not len(shape):
            raise ValueError("Either z_init or shape must be provided")
        
        if z_init is None:
            noise_scale = self.shift if noise_scale is None else noise_scale
            mu = self.mu if mu is None else mu
            z_init = torch.randn(shape, device=device) * noise_scale + mu
        shape = z_init.shape
        steps += 1  # include t=0
        x = z_init
        # Build a uniform schedule then apply the Flux/SD3 shift so that
        # more steps are spent in the high-noise region (near t=1) where
        # the velocity field is less straight.
        t_uniform = torch.linspace(self.n_steps, 0.0, steps, device=device) / self.n_steps
        t_vals = self._shift_timesteps(t_uniform, self.shift)
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
           
        # AB2 needs the previous step's velocity; seed it with None so the
        # first step falls back to an Euler warm-up automatically.
        prev_v = None

        for i in range(steps - 1):  # Stop one step before to avoid index error
            t = t_vals[i]
            t_next = t_vals[i + 1]
            # Per-step dt: distance along t for this interval (always positive).
            # With a shifted schedule the intervals are non-uniform, so we
            # compute dt from the actual t_vals rather than using a fixed scalar.
            dt = (t - t_next).item()

            # Use .item() so torch.full receives a Python scalar, not a 0-d
            # tensor. A 0-d tensor as fill_value triggers a value guard under
            # torch.compile causing a retrace for every unique timestep.
            t_batch = torch.full((x.shape[0],), fill_value=t.item(), device=device)
            t_batch = self.to_t_inds(t_batch)

            v = model_fn(x, t_batch)

            if self.sampling_method == "euler":
                x = self.euler_step(x, dt, v)

            elif self.sampling_method == "rk":
                t_batch_next = torch.full((x.shape[0],), fill_value=t_next.item(), device=device)
                t_batch_next = self.to_t_inds(t_batch_next)
                x = self.rk2_step(x, dt, v, t_batch_next, model_fn)

            elif self.sampling_method == "rk4":
                t_mid_val = (t.item() + t_next.item()) / 2.0
                t_batch_mid = torch.full((x.shape[0],), fill_value=t_mid_val, device=device)
                t_batch_mid = self.to_t_inds(t_batch_mid)
                t_batch_next = torch.full((x.shape[0],), fill_value=t_next.item(), device=device)
                t_batch_next = self.to_t_inds(t_batch_next)
                x = self.rk4_step(x, dt, v, t_batch_mid, t_batch_next, model_fn)

            elif self.sampling_method == "ab2":
                if prev_v is None:
                    # Euler warm-up for the very first step
                    x = self.euler_step(x, dt, v)
                else:
                    x = self.ab2_step(x, dt, v, prev_v)
                prev_v = v

            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")
            if return_traj:
                traj.append(x.cpu())
        self.model.train(training)  # restore original mode
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
        return_trj: bool = True,
        noise_scale: float = 1.0,
        mu: float = 0.0,
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
            noise_scale=noise_scale,
            mu=mu,
        )
        if return_trj:
            latents, traj = samples
            latents = (unscale_latent(latents)).to(device)
            traj = [unscale_latent(lat).to(device) for lat in traj]
            all_latents = torch.cat([latents] + traj)
        else:
            latents = samples
            latents = unscale_latent(latents)
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
