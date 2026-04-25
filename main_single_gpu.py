import os
import shutil
import time
import torch
import argparse
import traceback
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from train_utils import train_epoch, val_epoch, compute_fid_score
from utils import (
    get_vae,
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
    SCALE_CONSTANT,
)
from Diffusion import RFDiffusion
from single_gpu_utils import (
    load_single_gpu_model,
    load_optimizer_state_single_gpu,
    save_model_single_gpu,
    save_optimizer_single_gpu,
)
from opt import DualOpt


def main(args):
    assert torch.cuda.is_available(), "CUDA is required for training"
    set_seed()
    enable_perf_flags()

    device = torch.device("cuda:0")
    master_process = True

    train_df = pd.read_csv(os.path.join(args.data_dir_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir_path, "val.csv"))
    images_dir_pth = args.data_dir_path

    train_ds = CifarDataset(train_df, base_imgs_path=images_dir_pth, val=False)
    val_ds = CifarDataset(val_df, base_imgs_path=images_dir_pth, val=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_sz,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_sz,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

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
        resume_weights_dir = args.resume_local
        weights_path = os.path.join(resume_weights_dir, "model.pt")
        ema_path = os.path.join(resume_weights_dir, "ema.pt")
        opt_path = os.path.join(resume_weights_dir, "optim.pt")

        print(f"Resuming locally from: {resume_weights_dir}")
        os.makedirs(args.logs_save_dir, exist_ok=True)
        for log_name in ("train_logs.txt", "val_logs.txt", "fid_logs.txt"):
            src = os.path.join(resume_weights_dir, log_name)
            if os.path.exists(src):
                print(f"Resuming {log_name} from {src}")
                shutil.copy(src, os.path.join(args.logs_save_dir, log_name))

    elif args.resume_dir:
        download_dir = download_checkpoint_from_hf(
            "Muhammed164/Dit",
            args.resume_dir,
            args.dir_path_save,
        )
        print(f"Checkpoints downloaded to {download_dir}")

        downloaded_weights_dir = os.path.join(args.dir_path_save, args.resume_dir)
        weights_path = os.path.join(downloaded_weights_dir, "model.pt")
        ema_path = os.path.join(downloaded_weights_dir, "ema.pt")
        opt_path = os.path.join(downloaded_weights_dir, "optim.pt")

        print(f"Resuming with weights from: {weights_path}")

        os.makedirs(args.logs_save_dir, exist_ok=True)
        for log_name in ("train_logs.txt", "val_logs.txt", "fid_logs.txt"):
            src = os.path.join(downloaded_weights_dir, log_name)
            if os.path.exists(src):
                print(f"Resuming {log_name} from {src}")
                shutil.copy(src, os.path.join(args.logs_save_dir, log_name))

    vae = get_vae(model_id=args.vae_model_id, rank=0)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    if args.compile:
        if not hasattr(torch, "compile"):
            print("Warning: torch.compile not available (requires PyTorch >= 2.0). Skipping compilation.")
        else:
            vae = torch.compile(vae, mode=args.compile_mode, dynamic=True)
            print(f"VAE compiled with torch.compile (mode={args.compile_mode}, dynamic=True)")

    model_config = load_json(args.model_config, env_vars=False)
    print(f"Model config loaded from: {args.model_config}")

    diff_net = load_single_gpu_model(use_mp, mp_dtype, weights_path=weights_path, device=device, **model_config)
    num_params = sum(p.numel() for p in diff_net.parameters())
    print(f"Model parameters: {num_params / 1e6:.3f}M parameters")

    if ema_path is not None and os.path.exists(ema_path):
        ema_net = load_single_gpu_model(use_mp, mp_dtype, weights_path=ema_path, device=device, **model_config)
    else:
        ema_net = load_single_gpu_model(use_mp, mp_dtype, weights_path=None, device=device, **model_config)
        with torch.no_grad():
            ema_net.load_state_dict(diff_net.state_dict())

    for p in ema_net.parameters():
        p.requires_grad = False

    opt_config = load_json("opt_config.json", env_vars=False)
    if args.lr is not None:
        opt_config["lr"] = args.lr
        print(f"LR overridden via --lr: {args.lr}")
    if args.warmup_steps is not None:
        opt_config["warmup_steps"] = args.warmup_steps
        print(f"Warmup steps overridden via --warmup-steps: {args.warmup_steps}")

    model_for_opt = diff_net
    if args.use_moun:
        optim = DualOpt(model_for_opt, lr=opt_config["lr"], weight_decay=opt_config["weight_decay"])
    else:
        param_groups = build_adamw_param_groups(
            model_for_opt, weight_decay=opt_config["weight_decay"], verbose=True
        )
        betas = tuple(opt_config.get("betas", (0.9, 0.95)))
        optim = torch.optim.AdamW(
            param_groups,
            lr=opt_config["lr"],
            betas=betas,
            fused=torch.cuda.is_available(),
        )

    if opt_path is not None and os.path.exists(opt_path):
        load_optimizer_state_single_gpu(optim, opt_path)
        for param_group in optim.param_groups:
            param_group["lr"] = opt_config["lr"]
            if "initial_lr" in param_group:
                param_group["initial_lr"] = opt_config["lr"]

    scheduler = None
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    cur_steps = steps_per_epoch * args.cur_epochs

    if args.use_scheduler:
        last_step = cur_steps - 1
        warmup_steps = int(opt_config.get("warmup_steps", 0))
        if args.scheduler_type == "cosine":
            scheduler = build_warmup_cosine_scheduler(
                optim,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr=opt_config.get("min_lr", 8e-6),
                last_step=last_step,
            )
        elif args.scheduler_type == "step":
            scheduler = StepLR(
                optim,
                step_size=args.scheduler_step_size,
                gamma=args.scheduler_gamma,
                last_epoch=last_step,
            )
        print(
            f"Using {args.scheduler_type} scheduler (step-based, {total_steps} total steps, "
            f"warmup={warmup_steps}, resuming from step {max(cur_steps, 0)})"
        )

    diff_net_for_save = diff_net
    ema_net_for_save = ema_net

    if args.compile:
        if hasattr(torch, "compile"):
            print(f"Compiling diff_net and ema_net with torch.compile (mode={args.compile_mode}, dynamic=True)...")
            diff_net = torch.compile(diff_net, mode=args.compile_mode, dynamic=True)
            ema_net = torch.compile(ema_net, mode=args.compile_mode, dynamic=True)
            print("Models compiled successfully")

    rf_sch_config = load_json("rc_sch_config.json", env_vars=False)
    diff = RFDiffusion(
        model=diff_net,
        sigma=rf_sch_config["sigma"],
        mu=rf_sch_config["mu"],
        n_steps=rf_sch_config["sample_steps"],
        sampler_dist=rf_sch_config["sampler_dist"],
        sampling_method=rf_sch_config["sampling_method"],
        vae=vae,
    )

    logs_save_dir = args.logs_save_dir
    save_dir_path = args.dir_path_save

    os.makedirs(save_dir_path, exist_ok=True)
    os.makedirs(logs_save_dir, exist_ok=True)

    train_logs_path = os.path.join(logs_save_dir, "train_logs.txt")
    val_logs_path = os.path.join(logs_save_dir, "val_logs.txt")
    fid_logs_path = os.path.join(logs_save_dir, "fid_logs.txt")

    torch.cuda.empty_cache()

    best_val_loss = float("inf")

    for e in range(args.cur_epochs, args.epochs):
        st = time.time()

        print(f"Started Training on: {e+1} / {args.epochs} epoch")
        if scheduler is not None:
            print(f"Current LR: {scheduler.get_last_lr()[0]:.6e}")

        global_step_start = e * steps_per_epoch
        train_losses = train_epoch(
            diff,
            train_loader,
            optim,
            0,
            1.0,
            ema_net,
            args.loss_type,
            ema_decay=opt_config.get("ema_decay", 0.9999),
            ema_warmup_steps=int(opt_config.get("ema_warmup_steps", 2000)),
            mp_dtype=mp_dtype,
            scheduler=scheduler,
            use_fsdp=False,
            global_step_start=global_step_start,
        )

        torch.cuda.empty_cache()

        val_losses = val_epoch(
            diff,
            val_loader,
            0,
            args.loss_type,
            mp_dtype=mp_dtype,
            distributed=False,
        )

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
        is_best = False

        epoch_time = time.time() - st
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0

        print(f"Epoch {e + 1} completed in {epoch_time:.2f} seconds")
        print(f"Average Train Loss: {avg_train_loss:.6f}")
        print(f"Average Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            is_best = True
            print(f">>>>>> New Best Val Loss: {best_val_loss:.6f}")

        with open(val_logs_path, "a") as f:
            for val_loss in val_losses:
                f.write(f"{val_loss:.6f}\n")

        with open(train_logs_path, "a") as f:
            for train_loss in train_losses:
                f.write(f"{train_loss:.6f}\n")

        if args.fid_freq > 0 and (e + 1) % args.fid_freq == 0:
            print(f"Computing FID at epoch {e + 1} (using EMA model)...")
            try:
                diff.set_model(ema_net)
                fid_score = compute_fid_score(
                    diff=diff,
                    val_ds=val_ds,
                    num_samples=args.num_fid_samples,
                    gen_steps=args.num_gen_steps,
                    latent_shape=(4, 32, 32),
                    device=device,
                    gen_labels=["bird", "cat", "dog"],
                    fid_batch_size=args.fid_batch_size,
                    vae_scale_factor=SCALE_CONSTANT,
                    mp_dtype=mp_dtype if use_mp else None,
                    rank=0,
                    fid_feature=args.fid_feature,
                    distributed=False,
                )
                if fid_score is not None:
                    print(f"FID at epoch {e + 1}: {fid_score:.4f}")
                    with open(fid_logs_path, "a") as f:
                        f.write(f"epoch_{e + 1}: {fid_score:.6f}\n")
            except Exception as ex:
                print(f"FID computation failed at epoch {e + 1}: {ex}")
                traceback.print_exc()
            finally:
                diff.set_model(diff_net)
                torch.cuda.empty_cache()

        should_save = (e + 1) % args.epoch_save_freq == 0 or (is_best and args.save_best)

        if should_save:
            print(f"Saving checkpoint at epoch {e+1}...")
            dir_path = os.path.join(args.dir_path_save, f"checkpoint_{e+1}")
            os.makedirs(dir_path, exist_ok=True)

            opt_path = os.path.join(dir_path, "optim.pt")
            ema_path = os.path.join(dir_path, "ema.pt")
            weights_path = os.path.join(dir_path, "model.pt")

            save_model_single_gpu(diff_net_for_save, weights_path)
            save_model_single_gpu(ema_net_for_save, ema_path)
            save_optimizer_single_gpu(optim, opt_path)

            for log_name in ("train_logs.txt", "val_logs.txt", "fid_logs.txt"):
                src = os.path.join(logs_save_dir, log_name)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(dir_path, log_name))

            samples_ema_path = os.path.join(dir_path, "ema_sample.png")

            print("Generating sample images...")
            try:
                # 4 samples per class → 12-image grid (4 columns × 3 rows)
                conditions_labels = ["bird", "cat", "dog"] * 4

                diff.set_model(ema_net)
                diff.model.eval()
                with torch.no_grad(), fixed_seed(42, device=device):
                    ema_samples = diff.generate(
                        args.num_gen_steps,
                        conditions_labels,
                        cfg_fac=args.sample_cfg_fac,
                        device=device,
                        latent_shape=(4, 32, 32),
                        return_trj=False,
                    )
                diff.set_model(diff_net)
                diff.model.train()

                ema_grid = images_to_grid(ema_samples, 4)
                ema_grid.save(samples_ema_path, format="PNG")
                print("Sample images saved")

                if args.push_hub:
                    print("Uploading to Hugging Face Hub...")
                    try:
                        upload_file_paths_to_hf(
                            [
                                opt_path,
                                ema_path,
                                weights_path,
                                samples_ema_path,
                                os.path.join(dir_path, "train_logs.txt"),
                                os.path.join(dir_path, "val_logs.txt"),
                                os.path.join(dir_path, "fid_logs.txt"),
                            ]
                        )
                    except Exception as ex:
                        print(f"Failed to upload to Hub: {ex}")

                print(f"Checkpoint saved to {dir_path}")
            except Exception as ex:
                print(f"Failed to generate samples: {ex}")
                traceback.print_exc()

        torch.cuda.empty_cache()
        print("-" * 60 + "\n")

    if master_process:
        print("Training completed successfully!")
        print(f"Best validation loss: {best_val_loss:.6f}")


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
    parser.add_argument("--num-gen-steps", type=int, default=64)
    parser.add_argument("--model-config", type=str, default="model_config.json")

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override the LR warmup steps from opt_config.json (0 disables warmup)")
    parser.add_argument("--resume-dir", type=str, default="", help="HF Hub checkpoint folder (download then resume)")
    parser.add_argument("--resume-local", type=str, default="", help="Local checkpoint directory (no download)")

    parser.add_argument("--compile", type=str_to_bool, default=False)
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )

    parser.add_argument("--use-scheduler", type=str_to_bool, default=True)
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "step"])
    parser.add_argument("--scheduler-step-size", type=int, default=100)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--save-best", type=str_to_bool, default=False)

    parser.add_argument("--sample-cfg-fac", type=float, default=4.0, help="CFG scale used when generating checkpoint sample images")

    parser.add_argument("--fid-freq", type=int, default=25)
    parser.add_argument("--num-fid-samples", type=int, default=256)
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid-feature", type=int, default=64, choices=[64, 192, 768, 2048])

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)

    main(parser.parse_args())
