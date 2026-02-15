from tqdm import tqdm
from Diffusion import RFDiffusion
import torch.distributed as dist
import torch
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


def train_epoch(
    diff: RFDiffusion,
    train_ds,
    opt,
    rank,
    max_norm=None,
    ema_model=None,
    loss_type="mse_loss",
    ema_decay=0.999,
    grad_accum_steps=1,
    mp_dtype=torch.float16,
):
    diff.model.train()
    losses = []
    loop = tqdm(train_ds, desc="Training") if rank == 0 else train_ds
    
    # Only use scaler for float16 to prevent underflow
    # bfloat16 has enough dynamic range and doesn't need scaling
    use_scaler = (mp_dtype == torch.float16)
    scaler = ShardedGradScaler() if use_scaler else None
    
    for i, (inputs, labels) in enumerate(loop):
        with torch.amp.autocast(device_type="cuda",dtype=mp_dtype):
            inputs = inputs.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)
            
            is_accumulating = (i + 1) % grad_accum_steps != 0
            
            # Control gradient sync for FSDP2
            if hasattr(diff.model, 'set_requires_gradient_sync'):
                diff.model.set_requires_gradient_sync(not is_accumulating)
            
            loss = diff.rectified_flow_loss(inputs, labels, loss_type=loss_type)
            loss = loss / grad_accum_steps  # Scale loss
            
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if not is_accumulating:
            if max_norm is not None:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(diff.model.parameters(), max_norm)
            
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
                
            opt.zero_grad(set_to_none=True)
            
            if ema_model is not None:
                update_ema_model_fsdp(diff.model, ema_model, ema_decay)
        
        loss_scalar = loss.detach() * grad_accum_steps
        dist.all_reduce(loss_scalar, op=dist.ReduceOp.AVG)
        
        if rank == 0:
            losses.append(loss_scalar.item())
            loop.set_postfix({"loss": f"{sum(losses)/len(losses):.4f}"})
    
    return losses if rank == 0 else []


@torch.no_grad()
def val_epoch(diff: RFDiffusion, val_ds, rank, loss_type="mse_loss"):
    diff.model.eval()
    losses = []
    loop = tqdm(val_ds, desc="Validation") if rank == 0 else val_ds
    
    for inputs, labels in loop:
        
        inputs = inputs.to(rank, non_blocking=True)
        labels = labels.to(rank, non_blocking=True)
        
        loss = diff.rectified_flow_loss(inputs, labels, loss_type=loss_type)
        
        # Synchronize loss across all ranks
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        
        # All ranks compute, but only rank 0 logs
        if rank == 0:
            losses.append(loss.item())
            loop.set_postfix({"val_loss": f"{sum(losses)/len(losses):.4f}"})
    
    return losses if rank == 0 else []


def update_ema_model_fsdp(model, ema_model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1 - decay)