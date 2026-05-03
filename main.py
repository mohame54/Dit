import os
import shutil
import time
import torch
import argparse
import traceback
import pandas as pd
from torch.distributed import init_process_group, destroy_process_group, barrier, broadcast
from torch.optim.lr_scheduler import StepLR
from train_utils import train_epoch, val_epoch, compute_fid_score
from utils import (
    get_vae,
    load_data_loader_ddp,
    load_json,
    CifarDataset,
    upload_file_paths_to_hf,
    images_to_grid,
    set_seed,
    fixed_seed,
    download_checkpoint_from_hf,
    build_adamw_param_groups,
    build_warmup_cosine_scheduler,
    enable_perf_flags,
    compute_dataset_stats,
)
from Diffusion import RFDiffusion
from fsdp_utils import ( 
    load_fsdp_model,
    load_optimizer_state_fsdp,
    save_model_fsdp,
    save_optimizer_fsdp,
    copy_fsdp_model_state,
)
from ddp_utils import (
    load_ddp_model,
    load_optimizer_state_ddp,
    save_model_ddp,
    save_optimizer_ddp
)
from opt import DualOpt


def cleanup():
    destroy_process_group()


def main(args):
    use_fsdp = args.dist_mode == "fsdp"
    assert torch.cuda.is_available(), "CUDA is required for distributed training"
    set_seed()
    enable_perf_flags()
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
    full_images_dir_path = os.path.join(args.data_dir_path, args.full_images_name)
    print(f"Full images directory path: {full_images_dir_path}")
    if os.path.exists(full_images_dir_path):
        print(f"Full images directory path exists: {full_images_dir_path}")
        scale_constant, dataset_mean = compute_dataset_stats(full_images_dir_path)
        print(f"Scale constant: {scale_constant}")
        print(f"Dataset mean: {dataset_mean}")
    else:
        raise FileNotFoundError(f"Full images directory path does not exist: {full_images_dir_path}")
    
    train_ds = CifarDataset(train_df, base_imgs_path=images_dir_pth, val=False)
    val_ds = CifarDataset(val_df, base_imgs_path=images_dir_pth, val=False)

    train_loader = load_data_loader_ddp(train_ds, ddp_world_size, ddp_rank, batch_size=args.train_batch_sz, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = load_data_loader_ddp(val_ds, ddp_world_size, ddp_rank, batch_size=args.val_batch_sz, shuffle=False, num_workers=args.num_workers)

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
    
    if args.resume_local:
        # Resume directly from a local directory — no HF download
        resume_weights_dir = args.resume_local
        weights_path = os.path.join(resume_weights_dir, "model.pt")
        ema_path = os.path.join(resume_weights_dir, "ema.pt")
        opt_path = os.path.join(resume_weights_dir, "optim.pt")

        if master_process:
            print(f"Resuming locally from: {resume_weights_dir}")
            os.makedirs(args.logs_save_dir, exist_ok=True)
            for log_name in ("train_logs.txt", "val_logs.txt", "fid_logs.txt"):
                src = os.path.join(resume_weights_dir, log_name)
                if os.path.exists(src):
                    print(f"Resuming {log_name} from {src}")
                    shutil.copy(src, os.path.join(args.logs_save_dir, log_name))

        barrier()

    elif args.resume_dir:
        if master_process:
            # Only master process downloads to avoid race conditions
            # usage: download_checkpoint_from_hf(repo_id, checkpoint_dir, local_dir)
            # This downloads repo_id/checkpoint_dir/* to local_dir/checkpoint_dir/*
            download_dir = download_checkpoint_from_hf(
                "Muhammed164/Dit", 
                args.resume_dir, 
                args.dir_path_save 
            )
            print(f"Checkpoints downloaded to {download_dir}")
        
        # Wait for master to finish downloading
        barrier()
        
        # All processes look for files in the same location
        # download_checkpoint_from_hf returns os.path.join(local_dir, checkpoint_dir)
        # so we reconstruct that path here using the same logic
        downloaded_weights_dir = os.path.join(args.dir_path_save, args.resume_dir)
        
        weights_path = os.path.join(downloaded_weights_dir, "model.pt")
        ema_path = os.path.join(downloaded_weights_dir, "ema.pt") 
        opt_path = os.path.join(downloaded_weights_dir, "optim.pt")
        
        if master_process:
            print(f"Resuming with weights from: {weights_path}")
        
        # Resume logs if they exist
        if master_process:
            os.makedirs(args.logs_save_dir, exist_ok=True)
            
            for log_name in ("train_logs.txt", "val_logs.txt", "fid_logs.txt"):
                src = os.path.join(downloaded_weights_dir, log_name)
                if os.path.exists(src):
                    print(f"Resuming {log_name} from {src}")
                    shutil.copy(src, os.path.join(args.logs_save_dir, log_name))

    # CRITICAL FIX: Load VAE on ALL ranks for distributed generation
    vae = get_vae(model_id=args.vae_model_id, rank=local_rank)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    
    if master_process:
        print("VAE loaded on all ranks for distributed generation")

    if args.compile:
        if not hasattr(torch, 'compile'):
            if master_process:
                print("Warning: torch.compile not available (requires PyTorch >= 2.0). Skipping VAE compilation.")
        else:
            # dynamic=True uses symbolic batch dims so different batch sizes
            # (train / val / generation / FID) never trigger a retrace.
            vae = torch.compile(vae, mode=args.compile_mode, dynamic=True)
            if master_process:
                print(f"VAE compiled with torch.compile (mode={args.compile_mode}, dynamic=True)")
    
    # Load model architecture config
    model_config = load_json(args.model_config, env_vars=False)
    if master_process:
        print(f"Model config loaded from: {args.model_config}")

    # Load model
    if use_fsdp:
        diff_net = load_fsdp_model(use_mp, mp_dtype, weights_path=weights_path, **model_config)
    else:
        diff_net = load_ddp_model(use_mp, mp_dtype, weights_path=weights_path, device=device, **model_config)
    
    if master_process:
        num_params = sum(p.numel() for p in diff_net.parameters())
        print(f"Model parameters: {num_params / 1e6:.3f}M parameters")
    
    # Create EMA model
    if use_fsdp:
        if ema_path is not None and os.path.exists(ema_path):
            ema_net = load_fsdp_model(use_mp, mp_dtype, weights_path=ema_path, **model_config)
        else:
            ema_net = load_fsdp_model(use_mp, mp_dtype, weights_path=None, **model_config)
            copy_fsdp_model_state(diff_net, ema_net)
    else:
        if ema_path is not None and os.path.exists(ema_path):
            ema_net = load_ddp_model(use_mp, mp_dtype, weights_path=ema_path, device=device, **model_config)
        else:
            ema_net = load_ddp_model(use_mp, mp_dtype, weights_path=None, device=device, **model_config)
            with torch.no_grad():
                ema_state = diff_net.module.state_dict()
                ema_net.module.load_state_dict(ema_state)
    
    for p in ema_net.parameters():
        p.requires_grad = False

    barrier()
    
    # Create optimizer AFTER diff_net is defined
    opt_config = load_json("opt_config.json", env_vars=False)
    if args.lr is not None:
        opt_config['lr'] = args.lr
        if master_process:
            print(f"LR overridden via --lr: {args.lr}")
    if args.warmup_steps is not None:
        opt_config['warmup_steps'] = args.warmup_steps
        if master_process:
            print(f"Warmup steps overridden via --warmup-steps: {args.warmup_steps}")
    # For DDP, the optimizer needs the unwrapped model's parameters
    model_for_opt = diff_net if use_fsdp else diff_net.module
    if args.use_moun:
        optim = DualOpt(model_for_opt, lr=opt_config['lr'], weight_decay=opt_config['weight_decay'])
    else:
        param_groups = build_adamw_param_groups(
            model_for_opt, weight_decay=opt_config['weight_decay'], verbose=master_process
        )
        betas = tuple(opt_config.get('betas', (0.9, 0.95)))
        optim = torch.optim.AdamW(
            param_groups,
            lr=opt_config['lr'],
            betas=betas,
            fused=torch.cuda.is_available(),
        )
   
    if opt_path is not None and os.path.exists(opt_path):
        if use_fsdp:
            load_optimizer_state_fsdp(diff_net, optim, opt_path)
        else:
            load_optimizer_state_ddp(optim, opt_path)
        # Reset LR to initial config value, otherwise scheduler starts from decayed value
        for param_group in optim.param_groups:
            param_group['lr'] = opt_config['lr']
            if "initial_lr" in param_group:
                param_group['initial_lr'] = opt_config['lr']
    
    # Create learning rate scheduler (step-based, not epoch-based)
    scheduler = None
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    cur_steps = steps_per_epoch * args.cur_epochs
    
    if args.use_scheduler:
        # last_epoch in PyTorch scheduler actually means last_step here since we step per optimizer step
        last_step = cur_steps - 1  # -1 because PyTorch uses -1 for "no steps done yet"
        warmup_steps = int(opt_config.get('warmup_steps', 0))

        if args.scheduler_type == "cosine":
            scheduler = build_warmup_cosine_scheduler(
                optim,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr=opt_config.get('min_lr', 8e-6),
                last_step=last_step,
            )
        elif args.scheduler_type == "step":
            scheduler = StepLR(optim, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma, last_epoch=last_step)

        if master_process:
            print(
                f"Using {args.scheduler_type} scheduler (step-based, {total_steps} total steps, "
                f"warmup={warmup_steps}, resuming from step {max(cur_steps, 0)})"
            )
    
    barrier()

    # Keep pre-compile references so FSDP/DDP save utilities always receive
    # the correctly-typed wrappers (torch.compile adds an extra layer).
    diff_net_for_save = diff_net
    ema_net_for_save = ema_net

    if args.compile:
        if not hasattr(torch, 'compile'):
            if master_process:
                print("Warning: torch.compile not available (requires PyTorch >= 2.0). Skipping model compilation.")
        else:
            if master_process:
                print(f"Compiling diff_net and ema_net with torch.compile (mode={args.compile_mode}, dynamic=True)...")
                print("  Tip: set TORCH_LOGS=recompiles to see any unexpected recompilations at runtime.")
            # dynamic=True: symbolic batch dimension prevents retracing when the
            # batch size changes across train / val / generation / FID calls.
            # train/eval mode switches still produce two separate cached graphs
            # (one each), which is unavoidable but only happens once per mode.
            diff_net = torch.compile(diff_net, mode=args.compile_mode, dynamic=True)
            ema_net = torch.compile(ema_net, mode=args.compile_mode, dynamic=True)
            if master_process:
                print("Models compiled successfully")

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
    fid_logs_path = os.path.join(logs_save_dir, "fid_logs.txt")
    
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
        global_step_start = e * steps_per_epoch
        train_losses = train_epoch(
            diff,
            train_loader,
            optim,
            local_rank,
            1.0,
            ema_net,
            args.loss_type,
            ema_decay=opt_config.get('ema_decay', 0.9999),
            ema_warmup_steps=int(opt_config.get('ema_warmup_steps', 2000)),
            mp_dtype=mp_dtype,
            scheduler=scheduler,
            use_fsdp=use_fsdp,
            global_step_start=global_step_start,
        )
        
        barrier()
        torch.cuda.empty_cache()
        
        # Validation phase
        val_losses = val_epoch(diff, val_loader, local_rank, args.loss_type, mp_dtype=mp_dtype)
                
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
        
        # FID computation (independent of checkpoint saving frequency)
        if args.fid_freq > 0 and (e + 1) % args.fid_freq == 0:
            if master_process:
                print(f"Computing FID at epoch {e + 1} (using EMA model)...")
            try:
                # Use the EMA model for FID — it produces cleaner samples and
                # gives a more reliable signal of true generative quality.
                diff.set_model(ema_net)
                # All ranks participate (generate() needs FSDP shards from all ranks)
                fid_score = compute_fid_score(
                    diff=diff,
                    val_ds=val_ds,
                    num_samples=args.num_fid_samples,
                    gen_steps=args.num_gen_steps,
                    latent_shape=(4, 32, 32),
                    device=device,
                    gen_labels=["bird", "cat", "dog"],
                    fid_batch_size=args.fid_batch_size,
                    mp_dtype=mp_dtype if use_mp else None,
                    rank=ddp_rank,
                    fid_feature=args.fid_feature,
                )
                if master_process and fid_score is not None:
                    print(f"FID at epoch {e + 1}: {fid_score:.4f}")
                    with open(fid_logs_path, "a") as f:
                        f.write(f"epoch_{e + 1}: {fid_score:.6f}\n")
            except Exception as ex:
                if master_process:
                    print(f"FID computation failed at epoch {e + 1}: {ex}")
                    traceback.print_exc()
            finally:
                diff.set_model(diff_net)
                barrier()
                torch.cuda.empty_cache()
        
        # Checkpoint saving - FSDP needs all ranks; DDP only rank 0
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
            
            dir_path = os.path.join(args.dir_path_save, f"checkpoint_{e+1}")
            opt_path = os.path.join(dir_path, "optim.pt")
            ema_path = os.path.join(dir_path, "ema.pt")
            weights_path = os.path.join(dir_path, "model.pt")
            
            if use_fsdp:
                save_model_fsdp(diff_net_for_save, weights_path)
                save_model_fsdp(ema_net_for_save, ema_path)
                save_optimizer_fsdp(diff_net_for_save, optim, opt_path)
            else:
                save_model_ddp(diff_net_for_save, weights_path)
                save_model_ddp(ema_net_for_save, ema_path)
                save_optimizer_ddp(optim, opt_path)
            
            # Save logs with checkpoint
            if master_process:
                for log_name in ("train_logs.txt", "val_logs.txt", "fid_logs.txt"):
                    src = os.path.join(logs_save_dir, log_name)
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(dir_path, log_name))
            
            barrier()  # Wait for all ranks to finish saving
            
            samples_ema_path = os.path.join(dir_path, "ema_sample.png")
            
            should_generate = use_fsdp or master_process
            
            if master_process:
                print(f"Generating sample images...")
            
            try:
                if should_generate:
                    # 4 samples per class → 12-image grid (4 columns × 3 rows)
                    conditions_labels = ["bird", "cat", "dog"] * 4

                    diff.set_model(ema_net)
                    diff.model.eval()
                    with torch.no_grad(), fixed_seed(42, device=device):
                        # fixed_seed ensures the same starting noise at every
                        # checkpoint for a clean apples-to-apples visual
                        # comparison across epochs.
                        ema_samples = diff.generate(
                            args.num_gen_steps,
                            conditions_labels,
                            cfg_fac=args.sample_cfg_fac,
                            device=device,
                            latent_shape=(4, 32, 32),
                            return_trj=False
                        )

                    diff.set_model(diff_net)
                    diff.model.train()

                    # Only master saves the results
                    if master_process:
                        ema_grid = images_to_grid(ema_samples, 4)
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
                                    os.path.join(dir_path, "train_logs.txt"),
                                    os.path.join(dir_path, "val_logs.txt"),
                                    os.path.join(dir_path, "fid_logs.txt"),
                                ])
                            except Exception as ex:
                                print(f"Failed to upload to Hub: {ex}")
                        
                        print(f"Checkpoint saved to {dir_path}")
                    
            except Exception as ex:
                if master_process:
                    print(f"Failed to generate samples: {ex}")
                    traceback.print_exc()
            
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
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes per GPU (increase if GPU idles waiting for data)")
    parser.add_argument("--push-hub", type=str_to_bool, default=True)
    parser.add_argument("--num-gen-steps", type=int, default=64, help="Number of diffusion sampling steps")
    parser.add_argument("--model-config", type=str, default="model_config.json", help="Path to the model architecture JSON config file")

    # Distributed mode
    parser.add_argument("--dist-mode", type=str, default="ddp", choices=["fsdp", "ddp"], help="Distributed training mode")
    
    # Scheduler arguments
    parser.add_argument("--use-scheduler", type=str_to_bool, default=True, help="Use learning rate scheduler")
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "step"], help="Type of scheduler")
    parser.add_argument("--scheduler-step-size", type=int, default=100, help="Step size for StepLR scheduler")
    parser.add_argument("--scheduler-gamma", type=float, default=0.5, help="Gamma for StepLR scheduler")
    
    # Checkpoint saving
    parser.add_argument("--save-best", type=str_to_bool, default=False, help="Save checkpoint when achieving best validation loss")

    # Resume training
    parser.add_argument("--resume-dir", type=str, help="Directory inside the HF Hub repo to download and resume from")
    parser.add_argument("--resume-local", type=str, default="", help="Local checkpoint directory to resume from (skips HF Hub download)")
    parser.add_argument("--lr", type=float, default=None, help="Override the learning rate from opt_config.json (useful when resuming)")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override the LR warmup steps from opt_config.json (0 disables warmup)")

    # torch.compile
    parser.add_argument("--compile", type=str_to_bool, default=False, help="Compile models with torch.compile for faster training and inference (requires PyTorch >= 2.0)")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode: 'default' balances compile time and speed, 'reduce-overhead' minimises kernel-launch overhead (good for fixed shapes), 'max-autotune' maximises throughput at the cost of a longer initial compile")

    # Sample generation
    parser.add_argument("--sample-cfg-fac", type=float, default=4.0, help="CFG scale used when generating checkpoint sample images")
    parser.add_argument("--full-images-name", type=str, default="latent_vectors", help="Noise scale used when generating checkpoint sample images")
    # FID evaluation
    parser.add_argument("--fid-freq", type=int, default=20, help="Compute FID every N epochs (0 = disabled)")
    parser.add_argument("--num-fid-samples", type=int, default=128, help="Number of real/fake image pairs for FID (>=2048 recommended)")
    parser.add_argument("--fid-batch-size", type=int, default=16, help="Batch size used when generating images for FID")
    parser.add_argument("--fid-feature", type=int, default=64, choices=[64, 192, 768, 2048],
                        help="Inception feature layer for FID. scipy sqrtm cost is O(n³): "
                             "64→instant, 192→fast, 768→slow, 2048→very slow. "
                             "Use 64 or 192 when num-fid-samples < 2048.")
    
    main(parser.parse_args())