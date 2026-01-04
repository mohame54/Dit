import torch
from torch import nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.checkpoint.state_dict import _init_optim_state
from model import get_model
from opt import DualOpt


def save_model_fsdp(model, path):
    sharded_sd = model.state_dict()
    cpu_state_dict = {}
    for param_name, sharded_param in sharded_sd.items():
        full_param = sharded_param.full_tensor()
        if torch.distributed.get_rank() == 0:
            cpu_state_dict[param_name] = full_param.cpu()
    torch.save(cpu_state_dict, path)


def save_optimizer_fsdp(opt, path):
    is_rank_zero = torch.distributed.get_rank() == 0
    def get_state(optimizer):
        sharded_sd = optimizer.state_dict()
        sharded_state = sharded_sd["state"]
        full_state = {}
        for group_id, sharded_group in sharded_state.items():
            group_state = {}
            for attr, sharded_tensor in sharded_group.items():
                if isinstance(sharded_tensor, DTensor):
                    # "exp_avg" in AdamW is `DTensor`
                    full_tensor = sharded_tensor.full_tensor()
                else:
                    # "step" in AdamW is plain tensor
                    full_tensor = sharded_tensor
                if is_rank_zero:
                    group_state[attr] = full_tensor.cpu()
        
            if is_rank_zero:
                full_state[group_id] = group_state 
    
        return {
                "param_groups": sharded_sd["param_groups"],
                "state": full_state,
            }
    if isinstance(opt, DualOpt):
        state = {
            "moun": get_state(opt.moun),
            "adam": get_state(opt.adam),
            "step": opt._step_counter
        }
    else:
        state = get_state(opt)
    torch.save(state, path)


def load_optimizer_state_fsdp(opt, opt_state_path):
    full_sd = torch.load(
        opt_state_path, mmap=True, weights_only=True, map_location="cpu"
    )

    def load_opt_state(full_sd):
        _init_optim_state(opt)
        param_groups = opt.state_dict()["param_groups"]
        state = opt.state_dict()["state"]

        full_param_groups = full_sd["param_groups"]
        full_state = full_sd["state"]

        for param_group, full_param_group in zip(param_groups, full_param_groups):
            for key, value in full_param_group.items():
                if key == "params":
                    continue
                param_group[key] = value

            for pid, full_pid in zip(param_group["params"], full_param_group["params"]):
                if pid not in state:
                    continue
                param_state = state[pid]
                full_param_state = full_state[full_pid]
                for attr, full_tensor in full_param_state.items():
                    sharded_tensor = param_state[attr]
                    if isinstance(sharded_tensor, DTensor):
                        # exp_avg is DTensor
                        param_state[attr] = distribute_tensor(
                            full_tensor,
                            sharded_tensor.device_mesh,
                            sharded_tensor.placements,
                        )
                    else:
                        # step is plain tensor
                        param_state[attr] = full_tensor
        return {
            "param_groups": param_groups,
            "state": state,
        }
    
    if isinstance(opt, DualOpt):
        moun_state = load_opt_state(full_sd["moun"])
        adam_state = load_opt_state(full_sd["adam"])
        state = {
            "moun": moun_state,
            "adam": adam_state,
            "step": full_sd["step"],
        }
        opt.load_state_dict(state)
    else:
        opt.load_state_dict(load_opt_state(full_sd))

    
def _load_sharded_model(model, weights_path):
    full_sd = torch.load(
            weights_path,
            mmap=True,
            weights_only=True,
            map_location='cpu',
    )
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        sharded_tensor = distribute_tensor(
            full_tensor,
            sharded_meta_param.device_mesh,
            sharded_meta_param.placements,
        )
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    model.load_state_dict(sharded_sd, assign=True)


def load_fsdp_model(use_mp, mp_dt, weights_path: str = None ,**model_kwargs):
    fsdp_kwargs = {}
    if use_mp:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=mp_dt,
            reduce_dtype=mp_dt,
        )
        
    model = get_model(**model_kwargs)

    for layer in model.blocks:
        fully_shard(layer, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)
    if weights_path:
        _load_sharded_model(model, weights_path)
   
    return model
