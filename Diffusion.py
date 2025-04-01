from scheduler import DDIMSampler
import numpy as np
import torch
from typing import Union, Optional, Tuple, List
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
import os
from PIL import Image


class DiffusionPipeline:
    def __init__(
        self,
        main_net,
        vae_net,
        ema_net = None,
        num_steps: int = 1000,
        input_res: Union[Tuple[int, int], List[int]] = (32, 32),
        emma: float = 0.999,
        label_dict = None,
        in_channel = 4,
        rank = None,
        noise_schedule_name="quad",
        **noise_sch_kwargs
    ):
        if rank is not None:
           self.device = rank
        else:
           self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps_net = main_net.to(self.device)
        self.ema_net = ema_net if ema_net is not None else copy.deepcopy(main_net)
        self.ema_net = self.ema_net.eval()
        self.vae = vae_net.eval().to(self.device)
        self.res = (in_channel,) + input_res if isinstance(input_res, tuple) else [in_channel] + input_res
        self.num_steps = num_steps
        self.prediction_type = noise_sch_kwargs.get("prediction_type", "vel")
        self.scheduler = DDIMSampler(
            schedule_name=noise_schedule_name,
            diff_train_steps=num_steps,
            **noise_sch_kwargs
        )
        self.emma = emma
        self.label_dict = label_dict
        self.sigmoid = lambda x: 1 / (1 + torch.exp(-x))

    @torch.inference_mode()
    def __call__(
        self,
        labels :Union[str, List[str], torch.Tensor],
        num_infer_steps :int = 20,
        cfg_fac :float = 1.0,
        pred_net: str = "ema",
        mode: str = "fp16",
        store=False,
        timesteps_mode="trailing"
    ):
         if pred_net == "eps":
            self.eval()

         if isinstance(labels, (str, list)):
            if isinstance(labels, str):
               labels = [labels]
            labels = [self.label_dict[l] + 1 for l in labels]
            labels = torch.tensor(labels, dtype=torch.int32, device=self.device)
         num_samples = int(labels.shape[0])
         latents = self.sample_loop(
            num_samples,
            num_infer_steps,
            pred_net=pred_net,
            labels=labels,
            cfg_fac=cfg_fac,
            mode=mode,
            store=store,
            timesteps_mode=timesteps_mode
         )
         if store:
          latents, latent_samples = latents
          latent_samples = [la *  (1/ 0.18215) for la in latent_samples]
         #latents = unormalize_latent(latents)
         samples = latents * (1/ 0.18215)
         if mode == "fp16":
           with torch.autocast("cuda", torch.float16):
                samples = self.vae.decode(samples).sample
           samples = samples.cpu()
         samples = samples * 0.5 + 0.5
         torch.cuda.empty_cache()
         if pred_net == "eps":
            self.train()
         if store:
            latent_samples = torch.cat(latent_samples)
            with torch.autocast("cuda", torch.float16):
                latent_samples = self.vae.decode(latent_samples).sample
            latent_samples = latent_samples * 0.5 + 0.5
            latent_samples = latent_samples.cpu()
            return samples, latent_samples
         return samples

    def get_model_fn(self, net_type="ema", inf_mode="fp16"):
        pred_net = getattr(self, net_type + "_net").eval()
        pred_net = pred_net.to(self.device)
        def fn(*args, **kwargs):
            if inf_mode == "fp16":
              with torch.autocast("cuda", torch.float16):
                e_t = pred_net(*args, **kwargs)
            else:
               e_t = pred_net(*args, **kwargs)
            return e_t
        return fn

    def sample_loop(
        self,
        num_samples: Optional[int] = 1,
        num_infer_steps: Optional[int] = 20,
        pred_net: Optional[str] = 'ema',
        labels: Optional[torch.Tensor] = None,
        cfg_fac: Optional[int] = 2.0,
        x_t: Optional[torch.Tensor] = None,
        mode: Optional[str] = "fp16",
        store: Optional[bool] = True,
        timesteps_mode="trailing"
    ):
        shape = (num_samples,) + self.res if isinstance(self.res, tuple) else [num_samples] + self.res
        x_t = torch.randn(*shape).to(self.device) if x_t is None else x_t
        model_fn = self.get_model_fn(pred_net, mode)
        self.scheduler.set_infer_steps(num_infer_steps, mode=timesteps_mode)
        if labels is not None:
           null_labels = torch.zeros_like(labels).to(labels.device)
        else:
          null_labels = torch.zeros((num_samples,), dtype=torch.int32).to(self.device)
        xts = []
        for step in range(num_infer_steps):
            t = int(self.scheduler.timesteps[step])
            if cfg_fac > 0.0:
               x_t_input = torch.cat([x_t] * 2)
               classes = torch.cat([null_labels, labels])
            else:
              x_t_input = x_t
              classes = labels if labels is not None else null_labels
            t_now = (torch.ones((x_t_input.shape[0],),
                            device=x_t.device,
                            dtype=torch.int32) * t).to(x_t.device)
            e_t = model_fn(x_t_input, t_now, labels=classes)
            if cfg_fac > 0.0:
              e_t_uncond, e_t_cond = e_t.chunk(2)
              e_t = e_t_uncond + cfg_fac * (e_t_cond - e_t_uncond)
            x_t, _ = self.scheduler.p_sample(x_t, t, e_t)
            if store:
               xts.append(x_t)
        if store:
           return x_t, xts
        return x_t


    def train_loss(
        self,
        input_batch: torch.Tensor,
        labels=None,
        loss_type: Optional[str] = 'mse_loss',
        timestamp_dist: Optional[str] = "normal",
        **losskwargs
    ):
        bs, _, _, _ = input_batch.shape
        # sample
        if timestamp_dist == "logit":
          t = torch.randn(bs)
          t = torch.sigmoid(t) 
          #w_t = t / (1 - t)
          t = (t* self.num_steps).int()
        else:
          t =  torch.randint(0, self.num_steps, size=(bs,))
          #w_t = torch.ones((input_batch.size(0),), dtype=input_batch.dtype)
        x_t, eps = self.scheduler.q_sample(input_batch, t)
        if self.prediction_type == "eps":
           tar = eps
        else:
           tar = self.scheduler.get_velocity(x_t, eps, t)
        t = t.to(self.device)
        eps_pred = self.eps_net(x_t, t, labels=labels)
        loss = getattr(torch.nn.functional, loss_type)(eps_pred, tar, **losskwargs)
        #loss = w_t * loss
        #loss = loss.mean()
        return loss

    def load_ema(self):
        self.ema_net.load_state_dict(self.eps_net.state_dict())

    def update_emma(self):
        is_ddp = isinstance(self.ema_net, DDP)
        if is_ddp:
            for p_ema, p in zip(self.ema_net.module.parameters(),self.eps_net.module.parameters()):
                p_ema.data = (1 - self.emma) * p.data + p_ema.data * self.emma
        else:
            for p_ema, p in zip(self.ema_net.parameters(),self.eps_net.parameters()):
                p_ema.data = (1 - self.emma) * p.data + p_ema.data * self.emma

    def train(self):
        self.eps_net.train()

    def eval(self):
        self.eps_net.eval()

    def parameters(self):
        return self.eps_net.parameters()

    def save(self,
             file_name: str):
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        ema_path = file_name + '/ema.pt'
        net_path = file_name + "/eps.pt"
        torch.save(self.ema_net.state_dict(), ema_path)
        torch.save(self.eps_net.state_dict(), net_path)

    def inv_transform(self, img):
        if len(img.shape) == 3:
          img = img.unsqueeze(0)
        img = img.cpu().permute(0, 2, 3, 1)
        imgs = []
        for i in range(img.size(0)):
          imgs.append(Image.fromarray((img[i].clamp(0, 1.0) * 255.0).numpy().astype(np.uint8)))
        return imgs


    def load(self,
             path_nets: str):
        pathes = [os.path.join(path_nets, p) for p in os.listdir(path_nets) if ("ema" in p or "eps" in p)]
        for index in range(len(pathes)):
            if "ema" in pathes[index]:
                break
        ema_p = pathes[index]
        eps_p = pathes[int(not index)]
        map_loc = 'cpu' if not torch.cuda.is_available() else 'cuda'
        self.eps_net.load_state_dict(torch.load(eps_p, map_location=map_loc))
        self.ema_net.load_state_dict(torch.load(ema_p, map_location=map_loc))
