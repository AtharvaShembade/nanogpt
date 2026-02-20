from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW

from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 #query heads
    n_kv_head: int = 6 #key/value heads
    n_embed: int = 768
    window pattern: str = "SSSL" 
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long


def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 #multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., :d]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embed = config.n_embed
        self.head_dim = self.n_embed // self.n_head
        assert self.n_embed % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embed, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embed, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embed, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)
        self.ve_gate_channels = 32
        self.ve_get = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_get(x[..., :self.ve_gate_channels])) # (B, T, n_kv_head), range (0, 2)

            v = v + gate.unsqueeze(-1) * ve

        # Apply rotatory embed
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, sin, cos), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )

            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y 
