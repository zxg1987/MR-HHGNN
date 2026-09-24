from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .node_level import NodeLevelAggregation
from .relations import (
    CROSS_RELATION_MAP,
    DEFAULT_RELATION_INDICES,
    MODALITIES,
    SELF_RELATION_MAP,
)


class PathLevelAggregation(nn.Module):

    def __init__(self, d: int = 256, num_paths: int = 2):
        super().__init__()
        self.num_paths = num_paths
        self.W_P = nn.Linear(d, d, bias=True)
        self.theta = nn.Parameter(torch.randn(d))
        nn.init.xavier_uniform_(self.W_P.weight)
        nn.init.normal_(self.theta, std=0.01)

    def forward(self, path_embeddings: List[torch.Tensor]) -> torch.Tensor:
        assert len(path_embeddings) == self.num_paths
        scores = torch.stack([
            (torch.tanh(self.W_P(h_r)) @ self.theta).mean()
            for h_r in path_embeddings
        ])
        beta = F.softmax(scores, dim=0)
        z = torch.zeros_like(path_embeddings[0])
        for r, h_r in enumerate(path_embeddings):
            z = z + beta[r] * h_r
        return z


class CrossModalAggregation(nn.Module):

    def __init__(self, d: int = 256, d_r: int = 256, num_modalities: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.d = d
        self.d_r = d_r
        self.M = num_modalities

        self.W_C = nn.Linear(2 * d + d_r, 1, bias=False)
        self.W_cm = nn.ModuleDict({m: nn.Linear(d, d, bias=False) for m in MODALITIES})
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_C.weight)
        for m in MODALITIES:
            nn.init.xavier_uniform_(self.W_cm[m].weight)

        self._relation_index_matrix = [
            [SELF_RELATION_MAP[i] if i == j else CROSS_RELATION_MAP[(i, j)]
             for j in MODALITIES]
            for i in MODALITIES
        ]

    def forward(self, z: Dict[str, torch.Tensor],
                relation_all: torch.Tensor) -> Dict[str, torch.Tensor]:
        T = z["time"].shape[0]
        d = self.d
        M = self.M
        device = z["time"].device

        z_stack = torch.stack([z[m] for m in MODALITIES], dim=0)
        transformed = torch.stack([self.W_cm[m](z[m]) for m in MODALITIES], dim=0)

        rel_indices = torch.tensor(self._relation_index_matrix, device=device,
                                   dtype=torch.long)
        rel_embeds = relation_all[rel_indices.view(-1)].view(M, M, -1)

        z_i_exp = z_stack.unsqueeze(1).expand(M, M, T, d)
        z_j_exp = z_stack.unsqueeze(0).expand(M, M, T, d)
        rel_exp = rel_embeds.unsqueeze(2).expand(M, M, T, -1)

        concat_all = torch.cat([z_i_exp, rel_exp, z_j_exp], dim=-1)
        scores = self.W_C(concat_all.reshape(-1, 2 * d + self.d_r)).view(M, M, T)
        beta = F.softmax(scores.permute(0, 2, 1), dim=-1)

        agg = torch.einsum("itj,jtd->itd", beta, transformed)
        agg = self.dropout(self.activation(agg))
        return {m: agg[i] for i, m in enumerate(MODALITIES)}


class HHGNNLayer(nn.Module):

    def __init__(self, d: int = 256, d_r: int = 256, num_heads: int = 4):
        super().__init__()
        self.node_agg_spatial = NodeLevelAggregation(d, d_r, num_heads)
        self.node_agg_temporal = NodeLevelAggregation(d, d_r, num_heads)
        self.path_agg = PathLevelAggregation(d, num_paths=2)
        self.cross_modal_agg = CrossModalAggregation(d, d_r, num_modalities=3)

    def forward(self, embeddings: Dict[str, torch.Tensor],
                adjacency: Dict[str, Dict[str, torch.Tensor]],
                relation_all: torch.Tensor,
                relation_indices: Dict[str, Dict[str, int]]) -> Dict[str, torch.Tensor]:
        path_outputs = {}
        for modal in MODALITIES:
            h = embeddings[modal]
            h_spatial = self.node_agg_spatial(
                h, adjacency["spatial"][modal],
                relation_all[relation_indices["spatial"][modal]])
            h_temporal = self.node_agg_temporal(
                h, adjacency["temporal"][modal],
                relation_all[relation_indices["temporal"][modal]])
            path_outputs[modal] = [h_spatial, h_temporal]

        z = {modal: self.path_agg(path_outputs[modal]) for modal in MODALITIES}
        return self.cross_modal_agg(z, relation_all)


class HHGNN(nn.Module):

    def __init__(self, d: int = 256, d_r: int = 256, num_heads: int = 4,
                 num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [HHGNNLayer(d, d_r, num_heads) for _ in range(num_layers)]
        )

    def forward(self, embeddings: Dict[str, torch.Tensor], adjacency: Dict,
                relation_all: torch.Tensor,
                relation_indices: Dict = DEFAULT_RELATION_INDICES) -> Dict[str, torch.Tensor]:
        z = embeddings
        for layer in self.layers:
            z = layer(z, adjacency, relation_all, relation_indices)
        return z
