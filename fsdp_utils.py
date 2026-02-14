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

from model import get_model, DitBlock
from opt import DualOpt


def save_model_fsdp(model, path):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    
    if torch.distributed.get_rank() == 0:
        torch.save(cpu_state, path)


def save_optimizer_fsdp(model, opt, path):
    """Save FSDP optimizer state using official state_dict approach."""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        if isinstance(opt, DualOpt):
            adam_state = FSDP.optim_state_dict(model, opt.adam)
            moun_state = {}
            if opt.moun_param_num > 0:
                moun_state = FSDP.optim_state_dict(model, opt.moun)
            
            full_state = {
                "adam": adam_state,
                "moun": moun_state,
                "step": opt._step_counter
            }
        else:
            full_state = FSDP.optim_state_dict(model, opt)
            
    if torch.distributed.get_rank() == 0:
        torch.save(full_state, path)


def load_optimizer_state_fsdp(model, opt, opt_state_path):
    """Load FSDP optimizer state using official state_dict approach."""
    full_sd = torch.load(
        opt_state_path, mmap=True, weights_only=True, map_location="cpu"
    )

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        if isinstance(opt, DualOpt):
            # Load Adam state
            sharded_adam = FSDP.optim_state_dict_to_load(model, opt.adam, full_sd["adam"])
            opt.adam.load_state_dict(sharded_adam)
            
            # Load Muon state if present
            if opt.moun_param_num > 0 and "moun" in full_sd:
                sharded_moun = FSDP.optim_state_dict_to_load(model, opt.moun, full_sd["moun"])
                opt.moun.load_state_dict(sharded_moun)
                
            opt._step_counter = full_sd.get("step", 0)
        else:
            sharded_state = FSDP.optim_state_dict_to_load(model, opt, full_sd)
            opt.load_state_dict(sharded_state)

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
        sync_module_states=True,  # Ensure consistent initialization across ranks
    )
    
    if use_mp:
        fsdp_kwargs["mixed_precision"] = MixedPrecision(
            param_dtype=mp_dt,
            reduce_dtype=torch.float32,  # Keep reductions in fp32 for stability
            buffer_dtype=mp_dt,
            cast_forward_inputs=True,
        )
        
    # Force initialization on CPU to avoid GPU memory spike before sharding
    # This ensures we don't have N full copies on GPU at startup
    with torch.device("cpu"):
        model = get_model(**model_kwargs)
    
    model = FSDP(model, **fsdp_kwargs)
    
    if weights_path:
        load_model_state_fsdp(model, weights_path)
   
    return model