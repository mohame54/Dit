import pandas as pd
import torch
import os
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import get_model
from diffusers.models import AutoencoderKL
from train_utils import train_epoch, val_epoch, create_opt
from utils import get_dataloader_ddp, load_json
from Diffusion import DiffusionPipeline


assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0
vars = load_json("config.json", env_vars=True)
EPOCHS = os.environ['EPOCHS']
CUR_EPOCH = os.environ['CUR_EPOCH']
EPOCHS_LOGS = os.environ['EPOCHS_LOGS']
ACCUM_STEPS =  os.environ['ACCUM_STEPS']
RANDOM_CFG =  os.environ['RANDOM_CFG']
USE_ADMW =  os.environ['USE_ADMW']
LOAD_WEIGHTS =  os.environ['LOAD_WEIGHTS']
LOSS_TYPE =  os.environ['LOSS_TYPE'] 
GLOBAL_TRAIN_BATCH_SIZE =  os.environ['GLOBAL_TRAIN_BATCH_SIZE']
TRAIN_BATCH_SIZE = GLOBAL_TRAIN_BATCH_SIZE // ddp_world_size
VAL_BATCH_SIZE = os.environ['VAL_BATCH_SIZE']
PATH_TO_SAVE = os.environ['PATH_TO_SAVE']

dd = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno":"spider",
}
rev_dd = {v: k for k, v in dd.items()}
label2id = dict(zip(dd.values(), range(len(dd))))

train_df= pd.read_csv("content/data/train.csv")
val_df = pd.read_csv("content/data/val.csv")
train_df['all_path'] = train_df['path'].apply(lambda x: os.path.join("content", x))
val_df['all_path'] = val_df['path'].apply(lambda x: os.path.join("content", x))
train_loader = get_dataloader_ddp(train_df, ddp_rank, ddp_world_size, batch_size=TRAIN_BATCH_SIZE)
val_loader = get_dataloader_ddp(val_df, ddp_rank, ddp_world_size, batch_size=VAL_BATCH_SIZE, shuffle=False, val=False)
diff_net = get_model(num_classes=len(label2id), patch_size=2).to(device)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

if USE_ADMW:
    optim = create_opt(diff_net, lr=5e-5)
else:
    optim = torch.optim.AdamW(diff_net.parameters(), lr=5e-5)


diff_net = DDP(diff_net, device_ids=[ddp_local_rank])
Model = DiffusionPipeline(
    main_net=diff_net,
    vae_net=vae,
    input_res=(32, 32),
    noise_schedule_name="cosv2",
    label_dict=label2id,
)


scaler = torch.amp.GradScaler()


torch.cuda.empty_cache()
for e in range(CUR_EPOCH, EPOCHS):
  st = time.time()
  if master_process:
    print(f"Started Training on: {e+1} / {EPOCHS}")
  train_loss = train_epoch(
                  Model,
                  train_loader,
                  optim,
                  rank=ddp_rank,
                  scaler=scaler,
                  loss_type=LOSS_TYPE,
                  max_norm=1.0,
                  update_emma=True,
                  grad_accum_steps=ACCUM_STEPS,
                  random_cfg=RANDOM_CFG)
  torch.cuda.empty_cache()
  if master_process:
      val_loss = val_epoch(
                    Model,
                    val_loader,
                    rank = ddp_rank,
                    loss_type=LOSS_TYPE,
                    random_cfg=RANDOM_CFG,
                 )
      with open("train_logs.txt", "a") as f:
            f.write(f"epoch:{e + 1} | train loss: {train_loss:.6f}| val loss:{val_loss:.6f}\n")
 
  if (e +1) % EPOCHS_LOGS  == 0  and master_process:
      os.makedirs(PATH_TO_SAVE, exist_ok=True)
      torch.save({"model":Model.eps_net.module.state_dict(), "optim":optim.state_dict()}, f"{PATH_TO_SAVE}/ddim_state.pt")
      torch.save({"model":Model.ema_net.module.state_dict()},f"{PATH_TO_SAVE}/ddim_emma.pt")
      
  torch.cuda.empty_cache()
  if master_process:  
    print("-" * 60 + "\n")
destroy_process_group()