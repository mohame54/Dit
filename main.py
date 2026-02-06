import os
import time
import torch
import argparse
import pandas as pd
from torch.distributed import init_process_group, destroy_process_group, barrier, broadcast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from train_utils import train_epoch, val_epoch
from utils import (
    get_vae,
    load_data_loader_ddp,
    load_json,
    CifarDataset,
    upload_file_paths_to_hf,
    images_to_grid
)
from Diffusion import RFDiffusion
from fsdp_utils import ( 
    load_fsdp_model,
    load_optimizer_state_fsdp,
    save_model_fsdp,
    save_optimizer_fsdp
)
from opt import DualOpt


def cleanup():
    """Cleanup distributed training."""
    destroy_process_group()


def main(args):
    assert torch.cuda.is_available(), "for now i think we need CUDA for FSDP"
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    ddp_rank = int(os.environ['RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{local_rank}')
    master_process = ddp_rank == 0
    torch.cuda.set_device(device)
    init_process_group(backend='nccl')

    # Synchronize after process group initialization
    barrier()

    train_df = pd.read_csv(os.path.join(args.data_dir_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir_path, "val.csv"))
    images_dir_pth = args.data_dir_path
    
    train_ds = CifarDataset(train_df, base_imgs_path=images_dir_pth, val=False)
    val_ds = CifarDataset(val_df, base_imgs_path=images_dir_pth, val=False)

    train_loader = load_data_loader_ddp(train_ds, ddp_world_size, ddp_rank, batch_size=args.train_batch_sz, shuffle=True)
    val_loader = load_data_loader_ddp(val_ds, ddp_world_size, ddp_rank, batch_size=args.val_batch_sz, shuffle=False)

    use_mp = args.mp_dt.lower() != "none"
    mp_dtype = None
    if use_mp:
        mp_dtype = getattr(torch, args.mp_dt)
    
    weights_dir_path = args.weights_dir_path if args.weights_dir_path != "" else None
    weights_path = None
    ema_path = None
    opt_path = None
    if weights_dir_path is not None:
        weights_path = os.path.join(weights_dir_path, "model.pt")
        ema_path = os.path.join(weights_dir_path, "ema.pt")
        opt_path = os.path.join(weights_dir_path, "optim.pt")

    vae = None
    if master_process:
        vae = get_vae(model_id=args.vae_model_id, rank=local_rank)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
    
    # Load model first
    diff_net = load_fsdp_model(use_mp, mp_dtype, weights_path=weights_path)
    
    # Create EMA model - FSDP doesn't support deepcopy, so we create a new instance
    if ema_path is not None and os.path.exists(ema_path):
        ema_net = load_fsdp_model(use_mp, mp_dtype, weights_path=ema_path)
    else:
        # Create a new FSDP model and copy the state dict from diff_net
        ema_net = load_fsdp_model(use_mp, mp_dtype, weights_path=None)
        with torch.no_grad():
            ema_state = diff_net.state_dict()
            ema_net.load_state_dict(ema_state, assign=True)
    
    for p in ema_net.parameters():
        p.requires_grad = False

    barrier()
    
    # Create optimizer AFTER diff_net is defined
    opt_config = load_json("opt_config.json", env_vars=False)
    if args.use_moun:
        optim = DualOpt(diff_net, lr=opt_config['lr'], weight_decay=opt_config['weight_decay'])
    else:
        optim = torch.optim.AdamW(diff_net.parameters(), lr=opt_config['lr'], weight_decay=opt_config['weight_decay'])
   
    if opt_path is not None and os.path.exists(opt_path):
        load_optimizer_state_fsdp(optim, opt_path)
    
    # Create learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        if args.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=opt_config.get('min_lr', 1e-6))
        elif args.scheduler_type == "step":
            scheduler = StepLR(optim, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        
        if master_process:
            print(f"Using {args.scheduler_type} scheduler")
    
    barrier()
    
    rf_sch_config = load_json("rc_sch_config.json", env_vars=False)
    diff = RFDiffusion(
        model=diff_net,
        sigma=rf_sch_config['sigma'],
        mu=rf_sch_config['mu'],
        n_steps=rf_sch_config['sample_steps'],
        sampler_dist=rf_sch_config['sampler_dist'],
        sampling_method=rf_sch_config['sampling_method'],
        vae=vae
    )

    logs_save_dir = args.logs_save_dir
    save_dir_path = args.dir_path_save
    
    if master_process:
        os.makedirs(save_dir_path, exist_ok=True)
        os.makedirs(logs_save_dir, exist_ok=True)
    
    barrier()
    
    train_logs_path = os.path.join(logs_save_dir, "train_logs.txt")
    val_logs_path = os.path.join(logs_save_dir, "val_logs.txt")
    
    torch.cuda.empty_cache()
    
    best_val_loss = float("inf")
    
    for e in range(args.cur_epochs, args.epochs):
        st = time.time()
        
        # Set epoch for distributed sampler BEFORE starting the epoch
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(e)
        
        if master_process:
            print(f"Started Training on: {e+1} / {args.epochs} epoch")
            if scheduler is not None:
                print(f"Current LR: {scheduler.get_last_lr()[0]:.6e}")
        
        # Training phase - all GPUs participate
        train_losses = train_epoch(diff, train_loader, optim, local_rank, 1.0, ema_net, args.loss_type)
        
        barrier()
        torch.cuda.empty_cache()
        
        # Validation phase
        val_losses = val_epoch(diff, val_loader, local_rank, args.loss_type)
        
        # Step the scheduler after validation
        if scheduler is not None:
            scheduler.step()
        
        barrier()
        
        # Calculate average losses
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
        is_best = False
        
        if master_process:
            epoch_time = time.time() - st
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
            
            print(f"Epoch {e + 1} completed in {epoch_time:.2f} seconds")
            print(f"Average Train Loss: {avg_train_loss:.6f}")
            print(f"Average Val Loss: {avg_val_loss:.6f}")
            
            # Track best validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                is_best = True
                print(f">>>>>> New Best Val Loss: {best_val_loss:.6f}")
            
            # Log validation losses
            with open(val_logs_path, "a") as f:
                for val_loss in val_losses:
                    f.write(f"{val_loss:.6f}\n")

            # Log training losses
            with open(train_logs_path, "a") as f:
                for train_loss in train_losses:
                    f.write(f"{train_loss:.6f}\n")
        
        # Synchronize is_best flag across all ranks to avoid deadlock
        is_best_tensor = torch.tensor([1.0 if is_best else 0.0], device=device)
        broadcast(is_best_tensor, src=0)
        is_best = bool(is_best_tensor.item())
        
        # Checkpoint saving - all ranks need to participate in FSDP state collection
        should_save = (e + 1) % args.epoch_save_freq == 0 or (is_best and args.save_best)
        
        if should_save:
            # Synchronize before saving
            if master_process:
                print(f"Saving checkpoint at epoch {e+1}...")
            barrier()
            
            # Create directory on master process
            if master_process:
                dir_path = os.path.join(args.dir_path_save, f"checkpoint_{e+1}")
                os.makedirs(dir_path, exist_ok=True)
            
            barrier()  # Wait for directory creation
            
            # All ranks need to participate in FSDP state collection
            dir_path = os.path.join(args.dir_path_save, f"checkpoint_{e+1}")
            opt_path = os.path.join(dir_path, "optim.pt")
            ema_path = os.path.join(dir_path, "ema.pt")
            weights_path = os.path.join(dir_path, "model.pt")
            
            save_model_fsdp(diff_net, weights_path)
            save_model_fsdp(ema_net, ema_path)
            save_optimizer_fsdp(optim, opt_path)
            
            barrier()  # Wait for all ranks to finish saving
            
            # CRITICAL FIX: All ranks must participate in sample generation
            # because FSDP models use collective operations (ALLGATHER)
            samples_eps_path = os.path.join(dir_path, "samples.png")
            samples_ema_path = os.path.join(dir_path, "ema_sample.png")
            
            if master_process:
                print(f"Generating sample images...")
            
            try:
                conditions_labels = ["bird", "cat", "dog"] * 2
                # All ranks must participate in generation (FSDP requirement)
                samples = diff.generate(args.num_gen_steps, conditions_labels, device=device, latent_shape=(4, 32, 32), return_trj=False)
                diff.set_model(ema_net)
                ema_samples = diff.generate(args.num_gen_steps, conditions_labels, device=device, latent_shape=(4, 32, 32), return_trj=False)
                diff.set_model(diff_net)
                
                # Only master process saves and uploads
                if master_process:
                    eps_grid = images_to_grid(samples, 3)
                    ema_grid = images_to_grid(ema_samples, 3)
                    eps_grid.save(samples_eps_path, format="PNG")
                    ema_grid.save(samples_ema_path, format="PNG")
                    print(f"Sample images saved")
                    
                    if args.push_hub:
                        print("Uploading to Hugging Face Hub...")
                        try:
                            upload_file_paths_to_hf([
                                opt_path,
                                ema_path,
                                weights_path,
                                samples_ema_path,
                                samples_eps_path
                            ])
                        except Exception as ex:
                            print(f"Failed to upload to Hub: {ex}")
                    
                    print(f"Checkpoint saved to {dir_path}")
            except Exception as ex:
                if master_process:
                    print(f"Failed to generate samples: {ex}")
            
            # Synchronize all ranks after generation
            barrier()
        
        torch.cuda.empty_cache()
        
        if master_process:  
            print("-" * 60 + "\n")
        
        barrier()
    
    if master_process:
        print("Training completed successfully!")
        print(f"Best validation loss: {best_val_loss:.6f}")
    
    barrier()
    cleanup()


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp-dt", type=str, default="float16")  
    parser.add_argument("--vae-model-id", type=str, default="stabilityai/sdxl-vae")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--cur-epochs", type=int, default=0) 
    parser.add_argument("--train-batch-sz", type=int, default=128)
    parser.add_argument("--val-batch-sz", type=int, default=16) 
    parser.add_argument("--use-moun", type=str_to_bool, default=True)
    parser.add_argument("--loss-type", type=str, default="mse_loss")
    parser.add_argument("--weights-dir-path", type=str, default="")
    parser.add_argument("--dir-path-save", type=str, default="checkpoints")
    parser.add_argument("--epoch-save-freq", type=int, default=20)
    parser.add_argument("--logs-save-dir", type=str, default="logs")
    parser.add_argument("--data-dir-path", type=str, default="data")
    parser.add_argument("--push-hub", type=str_to_bool, default=True)
    parser.add_argument("--num-gen-steps", type=int, default=64, help="Number of diffusion sampling steps")
    
    # Scheduler arguments
    parser.add_argument("--use-scheduler", type=str_to_bool, default=True, help="Use learning rate scheduler")
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "step"], help="Type of scheduler")
    parser.add_argument("--scheduler-step-size", type=int, default=100, help="Step size for StepLR scheduler")
    parser.add_argument("--scheduler-gamma", type=float, default=0.5, help="Gamma for StepLR scheduler")
    
    # Checkpoint saving
    parser.add_argument("--save-best", type=str_to_bool, default=False, help="Save checkpoint when achieving best validation loss")
    
    main(parser.parse_args())