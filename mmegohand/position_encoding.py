"""Learned positional encoding used by mmEgoHand."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PositionEmbeddingLearned(nn.Module):
    """Absolute 2D positional embedding from the original implementation."""

    def __init__(self, num_pos_feats: int = 15, max_size: int = 50) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(max_size, num_pos_feats)
        self.col_embed = nn.Embedding(max_size, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        if height > self.row_embed.num_embeddings:
            raise ValueError(f"height {height} exceeds positional limit")
        if width > self.col_embed.num_embeddings:
            raise ValueError(f"width {width} exceeds positional limit")

        columns = self.col_embed(torch.arange(width, device=x.device))
        rows = self.row_embed(torch.arange(height, device=x.device))
        pos = torch.cat(
            (
                columns.unsqueeze(0).repeat(height, 1, 1),
                rows.unsqueeze(1).repeat(1, width, 1),
            ),
            dim=-1,
        )
        return pos.permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
