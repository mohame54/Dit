import torch
from torch.utils.data import DataLoader, DistributedSampler
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset
from torchvision import transforms as T
import PIL.Image as Image
import numpy as np
from typing import Sequence, Optional
import matplotlib.pyplot as plt
import torch
import json
import os
from PIL import Image
from huggingface_hub import HfApi
from huggingface_hub.hf_api import HfFolder


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
        img = np.load(img_path) * 0.18215
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
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(rank)
    vae.eval()
    return vae


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
    return latent * 0.18215


def unscale_latent(latent):
    return latent * (1 /  0.18215)


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
