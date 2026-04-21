import torch

from model import get_model


def load_single_gpu_model(use_mp, mp_dt, weights_path: str | None = None, device=None, **model_kwargs):
    model = get_model(**model_kwargs)

    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    if device is not None:
        model = model.to(device)

    return model


def save_model_single_gpu(model, path: str) -> None:
    torch.save(model.state_dict(), path)


def save_optimizer_single_gpu(opt, path: str) -> None:
    torch.save(opt.state_dict(), path)


def load_optimizer_state_single_gpu(opt, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    opt.load_state_dict(state)
