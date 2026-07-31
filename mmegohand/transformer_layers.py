"""Transformer blocks used by the paper model.

The released checkpoint contains ``linear1`` and ``linear2`` parameters with a
2048-dimensional hidden layer. The original forward path instead uses ``ffn``
with a 1024-dimensional hidden layer. Both are retained by default so that the
state dict and the reported 293.11M parameter count remain reproducible.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import Tensor, nn


class FFN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class _CheckpointCompatibleLayer(nn.Module):
    def _register_legacy_ffn(
        self,
        d_model: int,
        legacy_ffn_dim: int,
        keep_legacy_parameters: bool,
        dropout: float,
    ) -> None:
        if keep_legacy_parameters:
            self.linear1 = nn.Linear(d_model, legacy_ffn_dim)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(legacy_ffn_dim, d_model)
        else:
            self.linear1 = None
            self.dropout = None
            self.linear2 = None

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos


class TransformerEncoderLayer(_CheckpointCompatibleLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        legacy_ffn_dim: int = 2048,
        active_ffn_dim: int = 1024,
        dropout: float = 0.1,
        keep_legacy_parameters: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self._register_legacy_ffn(
            d_model, legacy_ffn_dim, keep_legacy_parameters, dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = FFN(d_model, active_ffn_dim, d_model)

    def forward(self, src: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, need_weights=False)[0]
        return self.norm1(src + self.ffn(src2))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos)
        return self.norm(output) if self.norm is not None else output


class TransformerPoseDecoderLayer(_CheckpointCompatibleLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        legacy_ffn_dim: int = 2048,
        active_ffn_dim: int = 1024,
        dropout: float = 0.1,
        keep_legacy_parameters: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self._register_legacy_ffn(
            d_model, legacy_ffn_dim, keep_legacy_parameters, dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ffn = FFN(d_model, active_ffn_dim, d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            need_weights=False,
        )[0]
        return self.norm2(tgt + self.ffn(tgt2))


class TransformerPoseDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: TransformerPoseDecoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(
                    self.norm(output) if self.norm is not None else output
                )

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class TransformerContextDecoderLayer(_CheckpointCompatibleLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        legacy_ffn_dim: int = 2048,
        active_ffn_dim: int = 1024,
        dropout: float = 0.1,
        keep_legacy_parameters: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )
        self.multihead_attn2 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )
        self._register_legacy_ffn(
            d_model, legacy_ffn_dim, keep_legacy_parameters, dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ffn = FFN(d_model, active_ffn_dim, d_model)

    def forward(
        self,
        tgt: Tensor,
        pose_memory: Tensor,
        memory: Tensor,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn1(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            need_weights=False,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.multihead_attn2(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(pose_memory, pos),
            value=pose_memory,
            need_weights=False,
        )[0]
        return self.norm3(tgt + self.ffn(tgt2))


class TransformerContextDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: TransformerContextDecoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        pose_memory: Tensor,
        memory: Tensor,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(
                output + tgt,
                pose_memory,
                memory,
                pos=pos,
                query_pos=query_pos,
            )
            intermediate.append(
                self.norm(output) if self.norm is not None else output
            )
        return torch.stack(intermediate)


# Original checkpoints use these class names in some pickled metadata.
TransformerTemporalDecoderLayer = TransformerContextDecoderLayer
TransformerTemporalDecoder = TransformerContextDecoder


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])
