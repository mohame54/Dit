from dataclasses import dataclass
from torch import nn
import torch
import math


@dataclass
class DitConfig:
      freq_dim: int= 256
      hidden_dim:int = 768
      num_blocks:int = 12
      bias: bool = True
      qkv_bias:bool = False
      num_classes:int = 3  # bird, cat, dog
      tim_mlp_dim: int = 1
      max_freq:int = 10000
      num_heads:int = 8
      patch_size:int = 2
      mlp_fac: int = 4
      img_size:int = 32
      in_chs:int = 4
      out_chs:int = 4
      drop_rate:float = 0.2
      add_conv_final: bool = False
      add_norm_embd: bool = False
      use_gate_mlp: bool = True
      mlp_bias: bool = False
      mlp_dim: int = 2 * 768
      flip_sin_to_cos: bool = True
      label_drop_prob: float = 0.1


class DitTimeEmbd(nn.Module):
  def __init__(self, config: DitConfig):
     super(DitTimeEmbd, self).__init__()
     self.mlp = nn.Sequential(
         nn.Linear(config.freq_dim, config.hidden_dim * config.tim_mlp_dim, bias=config.bias),
         nn.SiLU(),
         nn.Linear(config.hidden_dim * config.tim_mlp_dim, config.hidden_dim, bias=config.bias),
     )
     use_cfg = int(config.label_drop_prob > 0.0)
     self.label_embd = nn.Embedding(config.num_classes + use_cfg, config.hidden_dim)
     self.max_freq = config.max_freq
     self.dim = config.freq_dim
     self.flip_sin_to_cos = config.flip_sin_to_cos
     self.label_dropout_prob = config.label_drop_prob

  def label_embedding(self, labels):
      if self.training:
         drop_ids = torch.rand(labels.size(0), device=labels.device) < self.label_dropout_prob
         labels = torch.where(drop_ids, 0, labels)
      labels = self.label_embd(labels)
      return labels

  @staticmethod
  def get_positional_encoding(t, dim, max_freq=10000, flip_sin_to_cos=True):
      half = dim // 2
      freqs = torch.exp(
          -math.log(max_freq) * torch.arange(start=0, end=half, dtype=torch.float32) / half
      ).to(device=t.device)
      args = t[:, None].float() * freqs[None]
      embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
      if flip_sin_to_cos:
        half_dim = dim // 2
        embedding = torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)
      if dim % 2 == 0:
          embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
      return embedding

  def forward(self, t, labels=None):
      t = self.get_positional_encoding(
        t, self.dim, self.max_freq, self.flip_sin_to_cos)
      # Cast to match the dtype of the model parameters (for mixed precision training)
      t = t.to(dtype=self.mlp[0].weight.dtype)
      t = self.mlp(t)
      if labels is not None:
        labels = self.label_embedding(labels)
        t = t + labels
      return t


class DitAttention(nn.Module):
  def __init__(self, config:DitConfig):
      super(DitAttention, self).__init__()
      self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, config.qkv_bias)
      self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, config.qkv_bias)
      self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, config.qkv_bias)
      self.wo = nn.Linear(config.hidden_dim, config.hidden_dim, config.qkv_bias)
      self.num_heads = config.num_heads
      self.head_dim = config.hidden_dim // self.num_heads
      self.attn_drop = config.drop_rate
      self.hidden_dim = config.hidden_dim

  def forward(self, hidden_states):
      bs = hidden_states.size(0)
      q = self.q_proj(hidden_states)
      k = self.k_proj(hidden_states)
      v = self.v_proj(hidden_states)
      q = q.view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
      k = k.view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
      v = v.view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
      attn_drop = self.attn_drop if self.training else 0.0
      attention_scores = nn.functional.scaled_dot_product_attention(
          q, k, v, dropout_p=attn_drop)

      attention_scores = attention_scores.permute(0, 2, 1, 3).reshape(bs, -1, self.num_heads * self.head_dim)
      return self.wo(attention_scores)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        config:DitConfig
    ):
        super().__init__()
        img_size = config.img_size
        patch_size = config.patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(config.in_chs, config.hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(config.hidden_dim, eps=1e-6) if config.add_norm_embd else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def modulate(hidden, scale, shift):
    return hidden * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaNorm(nn.Module):
  def __init__(self, config:DitConfig):
     super(AdaNorm, self).__init__()
     self.norm = nn.LayerNorm(config.hidden_dim, 1e-6, False)
     self.proj = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_dim, 6 * config.hidden_dim))

  def forward(self, hidden_states, cond):
      emb = self.proj(cond)
      shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
      hidden_states = modulate(self.norm(hidden_states), scale_msa, shift_msa)
      return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class DitMlp(nn.Module):
    def __init__(self, config: DitConfig):
        super(DitMlp, self).__init__()
        mlp_dim = int(config.mlp_fac * config.hidden_dim)
        self.fc1 = nn.Linear(config.hidden_dim, mlp_dim, bias=config.mlp_bias)
        self.fc2 = nn.Linear(mlp_dim, config.hidden_dim, bias=config.mlp_bias)
        if config.use_gate_mlp:
            self.gate_mlp = nn.Linear(config.hidden_dim, mlp_dim, bias=config.mlp_bias)
        else:
            self.gate_mlp = None
        self.act = nn.SiLU()
        self.use_gate_mlp = config.use_gate_mlp
    
    def gate(self, x):
        if self.use_gate_mlp:
            return self.gate_mlp(x)
        else:
            return 1.0

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x) * self.gate(x)))

class DitBlock(nn.Module):
  def __init__(self, config: DitConfig):
      super(DitBlock, self).__init__()
      self.norm1 = AdaNorm(config)
      self.norm2 = nn.LayerNorm(config.hidden_dim, 1e-6, False)
    
      self.attn = DitAttention(config)
      self.mlp = DitMlp(config)

  def forward(self, hidden_states, c):
      norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, c)
      attn_output = self.attn(norm_hidden_states)
      attn_output = gate_msa.unsqueeze(1) * attn_output
      hidden_states = hidden_states + attn_output
      hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(hidden_states), scale_mlp, shift_mlp))
      return hidden_states


class DitModel(nn.Module):
  def __init__(self, config:DitConfig):
      super(DitModel, self).__init__()
      self.config = config
      self.patch_embd = PatchEmbed(config)
      self.time_embd  = DitTimeEmbd(config)
      self.pos_embd = nn.Parameter(torch.randn(1, self.patch_embd.num_patches, config.hidden_dim))
      self.norm_final = nn.LayerNorm(config.hidden_dim, 1e-6, False)
      self.patch_size = config.patch_size
      self.out_ch = config.out_chs
      self.lin_final = nn.Linear(config.hidden_dim, config.out_chs *  self.patch_size * self.patch_size)
      if config.add_conv_final:
        self.conv = nn.Conv2d(config.out_chs, config.out_chs, kernel_size=1, bias=False)
      self.final_add_norm = nn.Sequential(
        nn.SiLU(),
        nn.Linear(config.hidden_dim, 2 * config.hidden_dim)
      )
      self.blocks = nn.ModuleList([])
      for _ in range(config.num_blocks):
        self.blocks.append(DitBlock(config))

      self.init_weights()

  def finalize(self, x, c):
      shift, scale = self.final_add_norm(c).chunk(2, dim=1)
      x = modulate(self.norm_final(x), scale, shift)
      return self.lin_final(x)

  def unpactchify(self, x):
      bs = x.size(0)
      h = int(x.size(1) ** 0.5)
      x = x.view(bs, h, h, self.patch_size, self.patch_size, self.out_ch)
      x = torch.einsum('nhwpqc->nchpwq', x)
      x = x.reshape(bs, -1, h * self.patch_size, h * self.patch_size)
      if self.config.add_conv_final:
        x = self.conv(x)
      return x

  def init_weights(self):
      def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
      self.apply(_basic_init)

  def forward(self, x, t, cond=None):
      x = self.patch_embd(x)
      x = x + self.pos_embd
      c = self.time_embd(t, cond)
      for blc in self.blocks:
        x = blc(x, c)
      x = self.finalize(x, c)
      return self.unpactchify(x)


def get_model(**kwargs):
    config = DitConfig(**kwargs)
    return DitModel(config)
