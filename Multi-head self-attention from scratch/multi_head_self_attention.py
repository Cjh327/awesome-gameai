from turtle import forward
from numpy import size
import torch
import torch.nn as nn
import torch.nn.functional as F


# Naive version
class MultiHeadSelfAttentionNaive(nn.Module):
    def __init__(self, embed_dim, n_heads):
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.n_heads = n_heads

        self.in_project = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_project = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        '''
        x: (B, N, C), C=embed_dim
        '''
        B, N, C = x.size()
        M = self.n_heads
        x = self.in_project(x) # (B, N, 3C)
        q, k, v = x[:, :, C], k[:, :, C:2*C], v[:, :, -C:] # (B, N, C)
        q = q.reshape(B, N, self.n_heads, -1).permute(0, 2, 1, 3) # (B, H, N, D)
        k = k.reshape(B, N, self.n_heads, -1).permute(0, 2, 1, 3) # (B, H, N, D)
        v = v.reshape(B, N, self.n_heads, -1).permute(0, 2, 1, 3) # (B, H, N, D)
        
        # attn = F.softmax(q @ k.transpose(-2, -1)) / torch.sqrt(self.head_dim) # (B, H, N, N)
        attn = F.softmax(q @ k.transpose(-2, -1)) # (B, H, N, N)
        x = (attn @ v).transpose(1, 2) # (B, N, H, D)
        x = x.reshape(B, N, -1) # (B, N, C)

        x = self.out_proj(x)


# Official version
class MultiHeadAttentionOfficial(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        '''
        (B, N, d_model) -> q: (B, N, n_head * d_k) 
                           k: (B, N, n_head * d_k)
                           v: (B, N, n_head * d_v)
                        -> attn: (B, n_head, N, N)
                           v: (B, n_head, N, d_v)
                        -> out: (B, n_head, N, d_v)
                        -> reshape: (B, N, n_head * d_v)
                        -> fc: (B, N, d_model)
                        -> dropout, residual, layer_norm
        '''
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, q, k, v):
        '''
        q: (B, N, d_model)
        k: (B, N, d_model)
        v: (B, N, d_model)
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        B, N = q.size(0), q.size(1)

        residual = q

        q = self.w_qs(q).view(B, N, n_head, d_k).transpose(1, 2)   # (B, n_head, N, d_k)
        k = self.w_qs(k).view(B, N, n_head, d_k).transpose(1, 2)   # (B, n_head, N, d_k)
        v = self.w_qs(v).view(B, N, n_head, d_v).transpose(1, 2)   # (B, n_head, N, d_v)

        attn = F.softmax(q @ k.transpose(2, 3)) # (B, n_head, N, N)
        q = (attn @ v).transpose(1, 2).contiguous().view(B, N, -1) # (B, N, n_head * d_v)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


