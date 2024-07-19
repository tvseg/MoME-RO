import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
from functools import partial
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat


class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, num_heads, stage, **kwargs):
        super(TPN_DecoderLayer, self).__init__(d_model=d_model, nhead=num_heads, **kwargs)
        del self.multihead_attn
        if stage == 2:
            self.multihead_attn = AttentionOT( 
                d_model, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        else:
            self.multihead_attn = Attention( 
                d_model, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt2, attn2 = self.multihead_attn(
            tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2


class AttentionOT(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.eps = 0.05 

    def forward(self, query, key, value):

        B, Nq, C = query.shape
        _, Nk, _ = key.shape

        xq = self.q(query)
        xk = self.k(key)
        v = self.v(value)
        
        xq = F.normalize(xq, dim=-1, p=2)
        xk = F.normalize(xk, dim=-1, p=2)

        # compute score map 
        sim = torch.einsum('bnc,bmc->bmn', xq, xk)
        sim = sim.contiguous().view(B, Nk, Nq) 
        wdist = 1.0 - sim
        xx = torch.zeros(B, Nk, dtype=sim.dtype, device=sim.device).fill_(1. / Nk)
        yy = torch.zeros(B, Nq, dtype=sim.dtype, device=sim.device).fill_(1. / Nq)

        T = Sinkhorn_log_exp_sum(wdist, xx, yy, self.eps)
        
        # T vs. T * sim
        score_map = (Nq * Nk * sim * T).view(B, Nk, Nq)  
        attn_save = score_map.clone().transpose(1, 2) # B Nk Nq
        attn = self.attn_drop(T)

        x = (attn.transpose(1, 2) @ v).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        attn_save = attn_save.reshape(B, Nq//2, 2, Nk)
        attn_save = attn_save.sum(dim=1) / (Nq//2)
        
        return x, attn_save


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):

        B, Nq, C = query.shape
        _, Nk, _ = key.shape

        q = self.q(query).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.permute(0, 1, 3, 2)) * self.scale
        attn_save = attn.clone() # B H Nk Nq
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        attn_save = attn_save.sum(dim=1) / self.num_heads
        attn_save = attn_save.reshape(B, Nq//2, 2, Nk)
        attn_save = attn_save.sum(dim=1) / (Nq//2)
        
        return x, attn_save
    
def Sinkhorn_log_exp_sum(C, mu, nu, epsilon=0.05):
    
    def _log_boltzmann_kernel(u, v, epsilon, C=None):
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= epsilon
        return kernel
  
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    thresh = 1e-3
    max_iter = 100
            
    for i in range(max_iter):
       
        u0 = u  # useful to check the update
        K = _log_boltzmann_kernel(u, v, epsilon, C)
        u_ = torch.log(mu + 1e-8) - torch.logsumexp(K, dim=2)
        u = epsilon * u_ + u
        
        K_t = _log_boltzmann_kernel(u, v, epsilon, C).permute(0, 2, 1).contiguous()
        v_ = torch.log(nu + 1e-8) - torch.logsumexp(K_t, dim=2)
        v = epsilon * v_ + v
        
        err = (u - u0).abs().mean()
        if err.item() < thresh:
            break
        
    K = _log_boltzmann_kernel(u, v, epsilon, C)
    T = torch.exp(K)
    print(i)

    return T

