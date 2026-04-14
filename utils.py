import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset
from torchvision import transforms as T
from typing import Sequence, Optional
import matplotlib.pyplot as plt
from huggingface_hub import HfApi
from huggingface_hub.hf_api import HfFolder


SCALE_CONSTANT = 0.13025

def load_hf_api():
   HfFolder.save_token(os.getenv("HF_TOKEN"))
   return HfApi()

def upload_file_paths_to_hf(pathes):
    api = load_hf_api()
    for pth in pathes:
         api.upload_file(
            path_or_fileobj= pth,
            path_in_repo =pth,
            repo_id="Muhammed164/Dit",
            repo_type="model"
        )


def download_checkpoint_from_hf(repo_id, checkpoint_dir, local_dir):
    from huggingface_hub import hf_hub_download
    
    os.makedirs(local_dir, exist_ok=True)
    
    required_files = ["model.pt"]
    optional_files = ["optim.pt", "ema.pt", "train_logs.txt", "val_logs.txt"]
    
    print(f"Downloading checkpoint from {repo_id}/{checkpoint_dir} to {local_dir}...")
    
    for filename in required_files:
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{checkpoint_dir}/{filename}",
            local_dir=local_dir,
        )
        print(f"Downloaded {filename}")

    for filename in optional_files:
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{checkpoint_dir}/{filename}",
                local_dir=local_dir,
            )
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Optional file {filename} not found, skipping: {e}")
            
    # hf_hub_download with local_dir mirrors the repo structure:
    # local_dir/checkpoint_dir/model.pt
    return os.path.join(local_dir, checkpoint_dir)

class CifarDataset(Dataset):
  def __init__(self, data_df, base_imgs_path="", val=False, img_sz=(256,256)):
      self.data_df = data_df
      self.val = val
      if val:
        self.val_transform = T.ToTensor()
      self.img_sz = img_sz
      self.base_imgs_path = base_imgs_path

  def __len__(self):
      return len(self.data_df)

  def __getitem__(self, idx):
      data = "latent_vector_path"
      img_path , label = self.data_df.iloc[idx][[data,"label"]]
      if img_path:
         img_path = os.path.join(self.base_imgs_path, img_path)
      if self.val:
         img = Image.open(img_path).convert("RGB").resize(self.img_sz, resample=Image.LANCZOS)
         img = self.val_transform(img)
      else:
        img = np.load(img_path)
        img = scale_latent(img)
        img = torch.from_numpy(img.squeeze())
      return img, label + 1


def load_data_loader_ddp(dataset, world_size, rank,batch_size, shuffle=True, num_workers=2, pin_memory=True):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def get_vae(model_id=None, rank=0):
    if model_id is None:
        model_id = "stabilityai/sd-vae-ft-mse"
    vae = AutoencoderKL.from_pretrained(model_id).to(rank)
    vae.eval()
    return vae


@torch.no_grad()
def visualize_dataset_latent(
    vae,
    source,
    indices=None,
    device="cpu",
    save_path=None,
    show=True,
    label_names=None,
):
    """
    Decode dataset latents through the VAE and display the results.

    Use this to verify that the VAE model and SCALE_CONSTANT are consistent:
      - Scaled latent stats should be roughly  mean≈0, std≈1
      - Decoded images should look like the original training images

    Args:
        vae:          AutoencoderKL model (any device).
        source:       One of —
                        • CifarDataset  → use `indices` to select samples
                        • str path to a raw .npy latent file (scale applied automatically)
                        • torch.Tensor shape (C,H,W) or (B,C,H,W), already scaled
        indices:      List of dataset indices to visualise (default: first 3).
        device:       Target device string, e.g. "cpu" or "cuda:0".
        save_path:    If given, saves the figure to this path.
        show:         Call plt.show() when True.
        label_names:  Optional dict {label_int → str} for subplot titles.
    """
    from torch.utils.data import Dataset as _Dataset

    vae = vae.to(device)
    vae.eval()

    # ── Collect (scaled_latent, title) pairs ─────────────────────────────────
    pairs = []

    if isinstance(source, str):
        raw = np.load(source).squeeze().astype(np.float32)
        latent = scale_latent(torch.from_numpy(raw))
        pairs.append((latent, "npy file"))

    elif isinstance(source, torch.Tensor):
        if source.dim() == 3:
            source = source.unsqueeze(0)
        for i in range(source.shape[0]):
            pairs.append((source[i].float(), f"tensor[{i}]"))

    elif isinstance(source, _Dataset):
        if indices is None:
            indices = list(range(min(3, len(source))))
        for idx in indices:
            latent, label_id = source[idx]
            label_id = int(label_id)
            if label_names and label_id in label_names:
                title = label_names[label_id]
            else:
                title = f"cls {label_id}"
            pairs.append((latent.float(), f"[{idx}] {title}"))

    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"VAE latent verification   SCALE_CONSTANT = {SCALE_CONSTANT}")
    print(sep)

    for ax, (latent, title) in zip(axes, pairs):
        # ── Diagnostics on the scaled latent ─────────────────────────────────
        mean_v  = latent.mean().item()
        std_v   = latent.std().item()
        min_v   = latent.min().item()
        max_v   = latent.max().item()
        print(f"\n  [{title}]  scaled latent  shape={tuple(latent.shape)}")
        print(f"    mean={mean_v:+.4f}  std={std_v:.4f}  "
              f"min={min_v:+.4f}  max={max_v:+.4f}")

        # ── Decode ────────────────────────────────────────────────────────────
        raw_latent = unscale_latent(latent).unsqueeze(0).to(device)
        decoded = vae.decode(raw_latent).sample          # (1, 3, H, W), ≈ [-1, 1]
        img = (decoded.squeeze(0).float() * 0.5 + 0.5).clamp(0.0, 1.0)
        img_np = img.permute(1, 2, 0).cpu().numpy()

        print(f"    decoded pixel  "
              f"min={img_np.min():.4f}  max={img_np.max():.4f}  "
              f"mean={img_np.mean():.4f}")

        # ── Plot ─────────────────────────────────────────────────────────────
        ax.imshow(img_np)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(
            f"μ={mean_v:+.2f}  σ={std_v:.2f}\nmin={min_v:+.2f}  max={max_v:+.2f}",
            fontsize=7,
        )
        ax.axis("off")

    print(f"\n  Expect: scaled latent  μ≈0, σ≈1  (confirms scale constant is right)")
    print(f"  Expect: decoded image looks like the original training image")
    print(f"{sep}\n")

    plt.suptitle(
        f"VAE Reconstruction Check  (SCALE_CONSTANT={SCALE_CONSTANT})",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def preprocess_image(image, resize=(256, 256)):
    image = image.resize(resize, resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def images_to_grid(images, rows):
    bs, C, H, W = images.shape
    
    cols = int(np.ceil(bs / rows))
    
    total_slots = rows * cols
    
    if bs < total_slots:
        num_padding = total_slots - bs
        padding = torch.ones(num_padding, C, H, W, dtype=images.dtype, device=images.device)
        images = torch.cat([images, padding], dim=0)
    
    images = torch.clamp(images, 0.0, 1.0)
    
    images = (images * 255).to(torch.uint8)
    
    images = images.cpu().numpy()
    
    images = images.reshape(rows, cols, C, H, W)
    
    # (rows, cols, C, H, W) -> (rows, H, cols, W, C)
    images = images.transpose(0, 3, 1, 4, 2)
    
    # Reshape to final grid: (rows * H, cols * W, C)
    grid_height = rows * H
    grid_width = cols * W
    images = images.reshape(grid_height, grid_width, C)
    return Image.fromarray(images)


def scale_latent(latent):
    return latent * SCALE_CONSTANT


def unscale_latent(latent):
    return latent * (1 / SCALE_CONSTANT)


def postprocess(img):
    return (img  + 1) * 0.5


def plot_grid_images(
    imgs,
    grid_shape: Sequence[int],
    titles: Optional[Sequence[str]] = None,
    fig_fac: int = 2,
) -> None:
    n_rows, n_cols = grid_shape
    plt.figure(figsize=(n_cols * fig_fac, n_rows * fig_fac))
    for row in range(n_rows):
        for col in range(n_cols):
            index = row * n_cols + col
            plt.subplot(n_rows, n_cols, index + 1)
            if titles is not None:
               plt.title(titles[index])
            img = imgs[index]
            plt.imshow(img)
            plt.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


def load_json(json_path, env_vars=True):
    json_vars = json.loads(open(json_path, "r").read())
    if env_vars:
        for k in json_vars:
            os.environ.setdefault(k, json_vars[k])
    return json_vars


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
