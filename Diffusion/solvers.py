import torch


def make_time_tensor(t: float, batch_size: int, device=None, dtype=None):
    return torch.full((batch_size, 1), float(t), device=device, dtype=dtype)


class ODSolversMixin:
    def euler_step(
        self,
        x: torch.Tensor,
        dt:float,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        x_next = x - dt* model_output
        return x_next
    
    def rk2_step(
        self,
        x: torch.Tensor,
        dt: float,
        model_output: torch.Tensor,
        t_2,
        model_fn,
    ) -> torch.Tensor:
        x_2 = x - dt * model_output
        model_output_2 = model_fn(x_2, t_2)
        x_next = x - (dt / 2) * (model_output + model_output_2)
        return x_next
        