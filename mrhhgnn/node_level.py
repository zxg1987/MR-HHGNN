import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeLevelAggregation(nn.Module):

    def __init__(self, d: int = 256, d_r: int = 256, num_heads: int = 4,
                 negative_slope: float = 0.2, dropout: float = 0.5):
        super().__init__()
        assert d % num_heads == 0
        self.d = d
        self.num_heads = num_heads
        self.d_k = d // num_heads
        self.scale = math.sqrt(self.d_k)

        self.W_Q_all = nn.Linear(d, d, bias=False)
        self.W_K_all = nn.Linear(d, d, bias=False)
        self.W_V_all = nn.Linear(d, d, bias=False)
        self.W_r_all = nn.Linear(d_r, d, bias=False)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = nn.ReLU(inplace=True)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor,
                e_r: torch.Tensor) -> torch.Tensor:
        T = h.shape[0]
        H = self.num_heads
        d_k = self.d_k

        Q = self.W_Q_all(h).view(T, H, d_k).permute(1, 0, 2)
        K = self.W_K_all(h).view(T, H, d_k).permute(1, 0, 2)
        V = self.W_V_all(h).view(T, H, d_k).permute(1, 0, 2)
        mu_r = self.W_r_all(e_r).view(H, 1, d_k)

        attn_scores = torch.bmm(Q * mu_r, K.transpose(1, 2)) / self.scale
        attn_scores = self.leaky_relu(attn_scores)
        attn_scores = attn_scores.masked_fill((adj == 0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=2).nan_to_num(0.0)
        out = torch.bmm(attn_weights, V)

        h_out = out.permute(1, 0, 2).contiguous().view(T, -1)
        h_out = self.activation(h_out + h)
        return self.out_dropout(h_out)


def add_self_loops(adj: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    return torch.clamp(adj + eye, max=1.0)
