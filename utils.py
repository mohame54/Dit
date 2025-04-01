import numpy as np
import torch
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset
from torchvision import transforms as T
import PIL.Image as Image
import numpy as np
from typing import Sequence, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import random
import os


class DataLoaderLite:
    def __init__(
        self,
        ds,
        batch_size,
        process_rank,
        num_processes,
        shuffle=False,
    ):
        self.batch_size = batch_size
        self.process_rank = process_rank
        self.num_processes = num_processes  
        self.data = ds 
        self.shuffle = shuffle
        self.reset()

    def __len__(self):
        total_batches = len(self.data) // self.batch_size
        return (total_batches + self.num_processes - 1) // self.num_processes  
    
    def reset(self):
        self.data_indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle (self.data_indices)

    def __getitem__(self, idx):
        global_idx = idx * self.num_processes + self.process_rank
        start_idx = global_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.data))

        if start_idx >= len(self.data):
            raise IndexError("Index out of range")

        imgs, labels = [], []
        for data_idx in self.data_indices[start_idx: end_idx]:
            img, label = self.data[data_idx]
            imgs.append(img)
            labels.append(torch.tensor(label))

        if end_idx >= len(self.data):
           # if the epoch ends we should reshuffle the data
           self.reset()     
        
        return torch.stack(imgs), torch.stack(labels)


class CifarDataset(Dataset):
  def __init__(self, data_df, val=False):
      self.data_df = data_df
      self.val = val
      self.val_transform = T.ToTensor()
  def __len__(self):
      return len(self.data_df)

  def __getitem__(self, idx):
      data = "all_path"
      img_path , label = self.data_df.iloc[idx][[data,"classes"]]
      if self.val:
         img = Image.open(img_path).convert("RGB").resize((256, 256), resample=Image.LANCZOS)
         img = self.val_transform(img)
      else:
        img = np.load(img_path) * 0.18215
        img = torch.from_numpy(img.squeeze())
      return img, label + 1


def get_vae(model_id=None, rank=0):
    if model_id is None:
        model_id = "stabilityai/sd-vae-ft-mse"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(rank)
    vae.eval()
    return vae


def get_dataloader_ddp(
    df,
    process_rank,
    num_processes,
    batch_size=64,
    shuffle=True,
    val=False,
):
    ds = CifarDataset(df, val)
    return DataLoaderLite(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_processes=num_processes,
        process_rank=process_rank
    )

def preprocess_image(image):
    #w, h = image.size
    #w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    #image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


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
