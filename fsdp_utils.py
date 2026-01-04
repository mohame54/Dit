import torch
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.checkpoint.state_dict import _init_optim_state
from model import get_model, DitBlock
from opt import DualOpt


def save_model_fsdp(model, path):
    """Save FSDP model using official state_dict approach."""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    
    if torch.distributed.get_rank() == 0:
        torch.save(cpu_state, path)


def save_optimizer_fsdp(opt, path):
    """Save FSDP optimizer state."""
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
    
    if is_rank_zero:
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

def load_model_state_fsdp(model, weights_path):
    """Load model state dict into FSDP model using official approach."""
    full_sd = torch.load(
        weights_path,
        mmap=True,
        weights_only=True,
        map_location='cpu',
    )
    
    # Use official FSDP state dict loading
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(full_sd)


def load_fsdp_model(use_mp, mp_dt, weights_path: str = None, sharding_strategy=ShardingStrategy.FULL_SHARD, **model_kwargs):
    """
    Load FSDP model with auto-wrap policy and proper configuration.
    
    Args:
        use_mp: Whether to use mixed precision
        mp_dt: Mixed precision dtype
        weights_path: Path to model weights
        sharding_strategy: FSDP sharding strategy (FULL_SHARD for ZeRO-3, SHARD_GRAD_OP for ZeRO-2)
        **model_kwargs: Additional model arguments
    """
    # Define auto-wrap policy for transformer blocks
    dit_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            DitBlock,
        },
    )
    
    fsdp_kwargs = dict(
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=dit_auto_wrap_policy,
        sharding_strategy=sharding_strategy,
    )
    
    if use_mp:
        fsdp_kwargs["mixed_precision"] = MixedPrecision(
            param_dtype=mp_dt,
            reduce_dtype=mp_dt,  # Keep reductions in fp32 for stability
            buffer_dtype=mp_dt,
            cast_model_outputs=True,
        )
        
    model = get_model(**model_kwargs)
    
    # Cast model to target dtype BEFORE FSDP wrapping
    # MixedPrecision policy doesn't auto-cast existing parameters
    if use_mp:
        model = model.to(dtype=mp_dt)
    
    model = FSDP(model, **fsdp_kwargs)
    
    if weights_path:
        load_model_state_fsdp(model, weights_path)
   
    return model
