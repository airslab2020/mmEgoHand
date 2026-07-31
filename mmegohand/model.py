"""Checkpoint-compatible mmEgoHand pose model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from .position_encoding import PositionEmbeddingLearned
from .transformer_layers import (
    TransformerContextDecoder,
    TransformerContextDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerPoseDecoder,
    TransformerPoseDecoderLayer,
)


@dataclass(frozen=True)
class ModelConfig:
    frames: int = 30
    radar_height: int = 256
    radar_width: int = 128
    hidden_dim: int = 512
    attention_heads: int = 8
    encoder_layers: int = 6
    pose_decoder_layers: int = 6
    context_decoder_layers: int = 30
    candidate_queries: int = 100
    keypoints_per_hand: int = 21
    legacy_ffn_dim: int = 2048
    active_ffn_dim: int = 1024
    dropout: float = 0.1
    radar_fusion_weight: float = 0.9
    imu_fusion_weight: float = 0.1
    keep_legacy_parameters: bool = True


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(
                [input_dim] + hidden, hidden + [output_dim]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, keypoints_per_hand: int) -> None:
        super().__init__()
        self.logitsclass_embed = nn.Linear(hidden_dim, 2)
        self.kpt_embed = MLP(
            hidden_dim, hidden_dim, keypoints_per_hand * 3, num_layers=3
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return self.logitsclass_embed(features), self.kpt_embed(features)


class MMEgoHandPose(nn.Module):
    """Estimate one- or two-hand 3D keypoints from radar and IMU clips.

    Input shapes:
        radar: ``[batch, 30, 256, 128]``
        imu: ``[batch, 30, 6]``

    Output shapes:
        pred_logits: ``[30, batch, 100, 2]``
        pred_kpt: ``[30, batch, 100, 63]``
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_frames = config.frames
        self.num_queries = config.candidate_queries

        self.mmwave_backbone = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=(3, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        self.fc_mmwave = nn.Linear(64 * 8 * 4, config.hidden_dim)
        self.imu_backbone = nn.LSTM(
            input_size=6,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        self.position_embedding = PositionEmbeddingLearned(num_pos_feats=15)

        encoder_layer = TransformerEncoderLayer(
            config.hidden_dim,
            config.attention_heads,
            config.legacy_ffn_dim,
            config.active_ffn_dim,
            config.dropout,
            config.keep_legacy_parameters,
        )
        self.encoder = TransformerEncoder(
            encoder_layer,
            config.encoder_layers,
            nn.LayerNorm(config.hidden_dim),
        )
        self.encoder_imu = TransformerEncoder(
            encoder_layer,
            config.encoder_layers,
            nn.LayerNorm(config.hidden_dim),
        )

        pose_layer = TransformerPoseDecoderLayer(
            config.hidden_dim,
            config.attention_heads,
            config.legacy_ffn_dim,
            config.active_ffn_dim,
            config.dropout,
            config.keep_legacy_parameters,
        )
        self.posedecoder = TransformerPoseDecoder(
            pose_layer,
            config.pose_decoder_layers,
            nn.LayerNorm(config.hidden_dim),
            return_intermediate=True,
        )

        context_layer = TransformerContextDecoderLayer(
            config.hidden_dim,
            config.attention_heads,
            config.legacy_ffn_dim,
            config.active_ffn_dim,
            config.dropout,
            config.keep_legacy_parameters,
        )
        self.temporaldecoder = TransformerContextDecoder(
            context_layer,
            config.context_decoder_layers,
            nn.LayerNorm(config.hidden_dim),
        )

        self.posequery_embed = nn.Embedding(config.frames, config.hidden_dim)
        self.temporalquery_embed = nn.Embedding(
            config.candidate_queries, config.hidden_dim
        )
        self.prediction_head = PredictionHead(
            config.hidden_dim, config.keypoints_per_hand
        )

        # Retained only for exact compatibility with the released paper model.
        if config.keep_legacy_parameters:
            self.input_proj = nn.Conv2d(
                config.hidden_dim, config.hidden_dim, kernel_size=1
            )
            self.feature_linear = nn.Linear(
                config.radar_height * config.radar_width, config.hidden_dim
            )
            self.feature_linear_imu = nn.Linear(6, config.hidden_dim)
        else:
            self.input_proj = None
            self.feature_linear = None
            self.feature_linear_imu = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _validate_inputs(self, radar: Tensor, imu: Tensor) -> None:
        expected_radar = (
            self.config.frames,
            self.config.radar_height,
            self.config.radar_width,
        )
        if radar.ndim != 4 or tuple(radar.shape[1:]) != expected_radar:
            raise ValueError(
                f"radar must have shape [B, {expected_radar[0]}, "
                f"{expected_radar[1]}, {expected_radar[2]}], got {tuple(radar.shape)}"
            )
        if imu.ndim != 3 or tuple(imu.shape[1:]) != (self.config.frames, 6):
            raise ValueError(
                f"imu must have shape [B, {self.config.frames}, 6], "
                f"got {tuple(imu.shape)}"
            )

    def forward(self, radar: Tensor, imu: Tensor) -> Dict[str, Tensor]:
        self._validate_inputs(radar, imu)
        batch_size = radar.shape[0]

        radar_features = self.mmwave_backbone(radar.unsqueeze(1))
        radar_features = radar_features.permute(0, 2, 1, 3, 4).contiguous()
        radar_features = self.fc_mmwave(
            radar_features.view(batch_size, self.config.frames, -1)
        )
        imu_features, _ = self.imu_backbone(imu)

        radar_pos = self.position_embedding(
            radar_features.reshape(batch_size, self.config.frames, 32, 16)
        )
        radar_pos = radar_pos.flatten(2).permute(1, 0, 2)
        imu_pos = self.position_embedding(
            imu_features.reshape(batch_size, self.config.frames, 32, 16)
        )
        imu_pos = imu_pos.flatten(2).permute(1, 0, 2)
        fused_pos = (
            self.config.radar_fusion_weight * radar_pos
            + self.config.imu_fusion_weight * imu_pos
        )

        radar_features = radar_features.permute(1, 0, 2)
        imu_features = imu_features.permute(1, 0, 2)
        radar_memory = self.encoder(radar_features, pos=radar_pos)
        imu_memory = self.encoder_imu(imu_features, pos=imu_pos)
        fused_memory = (
            self.config.radar_fusion_weight * radar_memory
            + self.config.imu_fusion_weight * imu_memory
        )

        pose_queries = self.posequery_embed.weight.unsqueeze(1).repeat(
            1, batch_size, 1
        )
        context_queries = self.temporalquery_embed.weight.unsqueeze(1).repeat(
            1, batch_size, 1
        )
        pose_features = self.posedecoder(
            torch.zeros_like(pose_queries),
            fused_memory,
            pos=fused_pos,
            query_pos=pose_queries,
        )
        context_features = self.temporaldecoder(
            torch.zeros_like(context_queries),
            pose_features[-1],
            fused_memory,
            pos=fused_pos,
            query_pos=context_queries,
        )
        pred_logits, pred_kpt = self.prediction_head(
            context_features.transpose(1, 2)
        )
        return {"pred_logits": pred_logits, "pred_kpt": pred_kpt}


# Backward-compatible aliases for checkpoints and downstream scripts.
HMTransformer = MMEgoHandPose
mmVR_Transformer = MMEgoHandPose


def build_model(config: ModelConfig | None = None) -> MMEgoHandPose:
    return MMEgoHandPose(config or ModelConfig())
