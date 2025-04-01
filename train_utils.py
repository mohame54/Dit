import torchvision.transforms.functional as F_vis
import torch
import inspect
import random
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP




# preprocess the data before KID
def prepare_kid(real,pred):
    real = F_vis.resize(real.clamp(0,1),(299,299))
    pred = F_vis.resize(pred.clamp(0,1),(299,299))
    return real,pred


def create_opt(model, weight_decay=0.1,lr=1e-4,betas=[0.9, 0.95], eps=1e-8):
    params_dict = {nm: p for nm, p in model.named_parameters() if p.requires_grad}
    to_decay = [ p for nm, p in params_dict.items() if p.dim() >=2 ]
    no_decay = [ p for nm, p in params_dict.items() if p.dim() <2 ]
    groups = [
        {"params": to_decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]
    fused = "fused" in inspect.signature(torch.optim.Adam).parameters
    optim = torch.optim.AdamW(groups, lr, betas, eps=eps, fused=fused)
    print(f"to decay: {sum([p.numel() for p in to_decay])} parameters no decay: {sum([p.numel() for p in no_decay])} parmerers.")
    return optim


mean = lambda x:sum(x)/len(x)


def train_epoch(
    model,
    train_ds,
    opt,
    rank,
    scaler,
    loss_type="mse_loss",
    max_norm=None,
    update_emma=True,
    grad_accum_steps=1,
):
    model.train()
    losses = []
    loop = tqdm(train_ds, desc="Training loop") if rank == 0 else train_ds
    opt.zero_grad()
    loss_accum = 0.0

    for i, (inputs, labels) in enumerate(loop):
        inputs = inputs.to(rank)
        labels = labels.to(rank)
        if isinstance(model.eps_net, DDP):
           model.eps_net.require_backward_grad_sync = (i + 1) % grad_accum_steps == 0
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model.train_loss(inputs, labels, loss_type=loss_type, timestamp_dist="uni")
        scaler.scale(loss).backward()
        loss_accum += loss.detach()

        if (i + 1) % grad_accum_steps == 0:
            if max_norm is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            opt.zero_grad()

            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            if update_emma:
                model.update_emma()
            
            torch.cuda.empty_cache()
            if rank == 0:
                losses.append(loss_accum.item())
                loop.set_postfix({"Training loss": f"{mean(losses):.4f}"})
            loss_accum = 0.0

    if rank == 0:
        return mean(losses)


@torch.no_grad()
def val_epoch(
    model,
    val_ds,
    rank,
    loss_type="mse_loss",
):
    model.eval()
    loop = tqdm(val_ds, desc="Validation loop") if rank == 0 else val_ds
    losses = []

    for inputs, labels in loop:
        inputs = inputs.to(rank)
        labels = labels.to(rank)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model.train_loss(inputs, labels, loss_type=loss_type, timestamp_dist="uni")
        
        loss = loss.detach().item()

        if rank == 0:
            losses.append(loss)
            loop.set_postfix({"Validation loss": f"{mean(losses):.4f}"})

    if rank == 0:
        return mean(losses)
