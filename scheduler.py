import torch
import numpy as np
import math
from typing import Union, Optional


class DDIMSampler:
    def __init__(
        self,
        schedule_name: str,
        diff_train_steps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        rescale_snr=True,
        prediction_type="vel",
        thresholding=False,
    ):
        # Validate inputs
        assert diff_train_steps > 0, "diff_train_steps must be positive."
        assert 0 < beta_start < beta_end, "beta_start must be less than beta_end and greater than 0."

        # Compute beta schedule
        betas = get_beta_schedule(
            schedule_name,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=diff_train_steps,
        )
        self.beta_end = beta_end
        self.beta_start = beta_start
        if rescale_snr:
          betas = rescale_zero_terminal_snr(betas)
        self.betas = betas
        self.alpha = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.prediction_type = prediction_type
        self.thresholding = thresholding

        # Initialize timesteps and parameters
        self.num_train_steps = diff_train_steps
        self._num_inference_steps = 20
        self.timesteps = np.arange(0, diff_train_steps)[::-1]
        self.eta = 0.0  # Default deterministic DDIM sampling

    def set_infer_steps(self, num_steps: int, mode="leading"):
        assert num_steps > 0, "Number of inference steps must be positive."
        self._num_inference_steps = num_steps
        if mode == "leading":
            step_ratio = self.num_train_steps // num_steps
            timesteps = (np.arange(0, num_steps) * step_ratio).round()[::-1].astype(np.int64)
        else:
            step_ratio = self.num_train_steps / self._num_inference_steps
            timesteps = np.round(np.arange(self.num_train_steps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        self.timesteps = timesteps

    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        alpha_t = self.alpha_cumprod[timestep]
        alpha_prev = self.alpha_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=self.betas.device)
        beta_t = 1 - alpha_t
        beta_prev = 1 - alpha_prev
        return (beta_prev / beta_t) / (1 - alpha_t / alpha_prev)

    @staticmethod
    def threshold_sample(sample: torch.Tensor, threshold: float = 0.9956, max_clip: float = 1.0) -> torch.Tensor:
        batch_size, channels, height, width = sample.shape
        dtype = sample.dtype
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()

        sample = sample.view(batch_size, -1)
        abs_sample = sample.abs()

        s = torch.quantile(abs_sample, threshold, dim=1)
        s = torch.clamp(s, min=1, max=max_clip).unsqueeze(1)

        sample = torch.clamp(sample, -s, s) / s
        return sample.view(batch_size, channels, height, width).to(dtype)

    def p_sample(
        self,
        sample: torch.Tensor,
        t_now: Union[torch.Tensor, int],
        model_output: torch.Tensor,
    ):
        prev_timestep = max(t_now - self.num_train_steps // self._num_inference_steps, 0)
        alpha_t = self.alpha_cumprod[t_now]
        beta_prod_t = 1 - alpha_t
        alpha_prev = self.alpha_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=sample.device)

        if self.prediction_type == "eps":
            x0_t = (sample - beta_prod_t.sqrt() * model_output) / alpha_t.sqrt()
            eta_t = model_output
        elif self.prediction_type == "sample":
            x0_t = model_output
            eta_t = (sample - alpha_t.sqrt() * x0_t) / beta_prod_t.sqrt()
        elif self.prediction_type == "vel":
            x0_t = alpha_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            eta_t = alpha_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
        else:
            raise ValueError(f"Invalid prediction_type: {self.prediction_type}. Must be one of `eps`, `sample`, or `vel`.")

        if self.thresholding:
            x0_t = self.threshold_sample(x0_t)

        # Corrected variance calculation
        variance = self._get_variance(t_now, prev_timestep)
        sigma_t = self.eta * torch.clamp(variance, min=1e-8).sqrt()
        c2 = ((1 - alpha_prev) - sigma_t ** 2).sqrt()

        x_tminus = alpha_prev.sqrt() * x0_t + c2 * eta_t

        # Add noise only when eta > 0
        if self.eta > 0:
            eps = torch.randn_like(sample)
            x_tminus += sigma_t * eps

        return x_tminus, x0_t

    def q_sample(
        self,
        x_t: torch.Tensor,
        timesteps: Union[torch.Tensor, int],
        eps: Optional[torch.Tensor] = None
    ):

        alpha_t = self.alpha_cumprod[timesteps].to(timesteps.device)
        alpha_t = alpha_t.flatten().to(x_t.device)[:, None, None, None]
        eps = torch.randn(*list(x_t.shape)).to(x_t.device) if eps is None else eps
        x_t = alpha_t.sqrt() * x_t + (1 - alpha_t).sqrt() * eps
        return x_t, eps

    def get_velocity(
        self, sample: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        alphas_cumprod = self.alpha_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt()
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]).sqrt()
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


def rescale_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas

def get_beta_schedule(
    beta_schedule: str,
    *,
    beta_start: float,
    beta_end: float,
    num_diffusion_timesteps: int
):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float32,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start

    elif beta_schedule == "cosv2":
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), beta_end))
        betas = np.array(betas)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas)
