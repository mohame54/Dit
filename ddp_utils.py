import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import get_model
from opt import DualOpt


def load_ddp_model(use_mp, mp_dt, weights_path: str = None, device=None, **model_kwargs):
    model = get_model(**model_kwargs)
    
    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model = DDP(model, device_ids=[device])
    return model


def save_model_ddp(model, path):
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), path)


def save_optimizer_ddp(opt, path):
    if dist.get_rank() == 0:
        torch.save(opt.state_dict(), path)


def load_optimizer_state_ddp(opt, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    opt.load_state_dict(state)
