from contextlib import nullcontext
from tqdm import tqdm
from Diffusion import RFDiffusion
import torch.distributed as dist
import torch
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from utils import SCALE_CONSTANT


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
    scheduler=None,
    use_fsdp=True,
):
    diff.model.train()
    losses = []
    loop = tqdm(train_ds, desc="Training") if rank == 0 else train_ds
    
    # Only use scaler for float16 to prevent underflow
    # bfloat16 has enough dynamic range and doesn't need scaling
    use_scaler = (mp_dtype == torch.float16)
    if use_scaler:
        scaler = ShardedGradScaler() if use_fsdp else torch.amp.GradScaler()
    else:
        scaler = None
    
    for i, (inputs, labels) in enumerate(loop):
        inputs = inputs.to(rank, non_blocking=True)
        labels = labels.to(rank, non_blocking=True)
        
        is_accumulating = (i + 1) % grad_accum_steps != 0
        
        # Control gradient sync: DDP uses no_sync(), FSDP uses set_requires_gradient_sync
        if is_accumulating and not use_fsdp:
            sync_ctx = diff.model.no_sync()
        else:
            if hasattr(diff.model, 'set_requires_gradient_sync'):
                diff.model.set_requires_gradient_sync(not is_accumulating)
            sync_ctx = nullcontext()
        
        with sync_ctx:
            with torch.amp.autocast(device_type="cuda", dtype=mp_dtype):
                loss = diff.rectified_flow_loss(inputs, labels, loss_type=loss_type)
                loss = loss / grad_accum_steps
            
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
        
            if scheduler is not None:
                scheduler.step()
            
            if ema_model is not None:
                update_ema_model(diff.model, ema_model, ema_decay)
        
        loss_scalar = loss.detach() * grad_accum_steps
        dist.all_reduce(loss_scalar, op=dist.ReduceOp.AVG)
        
        if rank == 0:
            losses.append(loss_scalar.item())
            loop.set_postfix({"loss": f"{sum(losses)/len(losses):.4f}"})
    
    return losses if rank == 0 else []


@torch.no_grad()
def val_epoch(diff: RFDiffusion, val_ds, rank, loss_type="mse_loss", mp_dtype=None):
    diff.model.eval()
    losses = []
    loop = tqdm(val_ds, desc="Validation") if rank == 0 else val_ds
    
    for inputs, labels in loop:
        
        inputs = inputs.to(rank, non_blocking=True)
        labels = labels.to(rank, non_blocking=True)
        
        with torch.amp.autocast(device_type="cuda", dtype=mp_dtype) if mp_dtype is not None else nullcontext():
            loss = diff.rectified_flow_loss(inputs, labels, loss_type=loss_type)
        
        # Synchronize loss across all ranks
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        
        # All ranks compute, but only rank 0 logs
        if rank == 0:
            losses.append(loss.item())
            loop.set_postfix({"val_loss": f"{sum(losses)/len(losses):.4f}"})
    
    return losses if rank == 0 else []


def update_ema_model(model, ema_model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1 - decay)


@torch.no_grad()
def compute_fid_score(
    diff: RFDiffusion,
    val_ds,
    num_samples: int,
    gen_steps: int,
    latent_shape: tuple,
    device,
    gen_labels: list,
    fid_batch_size: int = 16,
    vae_scale_factor: float = SCALE_CONSTANT,
    mp_dtype=None,
    rank: int = 0,
) -> float | None:
    """
    Compute FID between decoded val latents (real) and model-generated images (fake).

    All ranks must call this function together when using FSDP, because the fake
    image generation involves the sharded model. Real image collection and the
    metric computation itself are rank-0 only.

    Returns the FID scalar on rank 0, None on all other ranks.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torch.utils.data import DataLoader as _DataLoader

    is_master = (rank == 0)
    diff.model.eval()

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=mp_dtype)
        if mp_dtype is not None
        else nullcontext()
    )

    # ── Real images (rank 0 only — VAE is a plain nn.Module, no NCCL involved) ──
    fid_metric = None
    if is_master:
        fid_metric = (
            FrechetInceptionDistance(feature=2048, normalize=True)
            .to(device)
            .set_dtype(torch.float64)
        )
        real_loader = _DataLoader(
            val_ds, batch_size=fid_batch_size, shuffle=False, num_workers=2
        )
        real_count = 0
        real_bar = tqdm(
            total=num_samples,
            desc="FID real images",
            unit="img",
            leave=False,
        )
        for latents, _ in real_loader:
            latents = latents.to(device)
            raw = latents / vae_scale_factor  # undo the SCALE_CONSTANT applied by the dataset
            with autocast_ctx:
                decoded = diff.vae.decode(raw).sample
            decoded = (decoded.float() * 0.5 + 0.5).clamp(0.0, 1.0)
            fid_metric.update(decoded, real=True)
            batch_n = decoded.shape[0]
            real_count += batch_n
            real_bar.update(batch_n)
            if real_count >= num_samples:
                break
        real_bar.close()

    # ── Fake images (all ranks participate — model may be FSDP-sharded) ────────
    labels_cycle = (gen_labels * (num_samples // len(gen_labels) + 1))[:num_samples]
    fake_count = 0
    gen_bar = tqdm(
        total=num_samples,
        desc="FID generated",
        unit="img",
        leave=False,
        disable=not is_master,   # only rank 0 shows the bar
    )
    for start in range(0, num_samples, fid_batch_size):
        batch_labels = labels_cycle[start : start + fid_batch_size]
        fake_imgs = diff.generate(
            gen_steps,
            batch_labels,
            device=device,
            latent_shape=latent_shape,
            return_trj=False,
            cfg_fac=2.0,
        )
        if is_master:
            fid_metric.update(fake_imgs.float().to(device), real=False)
        batch_n = len(batch_labels)
        fake_count += batch_n
        gen_bar.update(batch_n)
        if fake_count >= num_samples:
            break
    gen_bar.close()

    diff.model.train()

    if is_master:
        return fid_metric.compute().item()
    return None