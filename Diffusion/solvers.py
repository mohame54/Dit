import torch


def make_time_tensor(t: float, batch_size: int, device=None, dtype=None):
    return torch.full((batch_size, 1), float(t), device=device, dtype=dtype)


class ODSolversMixin:
    def euler_step(
        self,
        x: torch.Tensor,
        dt: float,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        return x - dt * model_output

    def rk2_step(
        self,
        x: torch.Tensor,
        dt: float,
        model_output: torch.Tensor,
        t_next,
        model_fn,
    ) -> torch.Tensor:
        """Heun's method (explicit trapezoidal rule): 2nd-order, 2 NFE/step."""
        x_2 = x - dt * model_output
        model_output_2 = model_fn(x_2, t_next)
        x_next = x - (dt / 2) * (model_output + model_output_2)
        return x_next

    def rk4_step(
        self,
        x: torch.Tensor,
        dt: float,
        model_output: torch.Tensor,
        t_mid,
        t_next,
        model_fn,
    ) -> torch.Tensor:
        k1 = model_output
        k2 = model_fn(x - (dt / 2) * k1, t_mid)
        k3 = model_fn(x - (dt / 2) * k2, t_mid)
        k4 = model_fn(x - dt * k3, t_next)
        return x - (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def ab2_step(
        self,
        x: torch.Tensor,
        dt: float,
        model_output: torch.Tensor,
        prev_model_output: torch.Tensor,
    ) -> torch.Tensor:
        return x - (dt / 2) * (3 * model_output - prev_model_output)
