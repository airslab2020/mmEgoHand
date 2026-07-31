"""Set-prediction losses and pose metrics."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .matcher import HungarianMatcher, Match


def move_targets(
    targets: Sequence[Dict[str, Tensor]], device: torch.device
) -> List[Dict[str, Tensor]]:
    return [
        {name: value.to(device) for name, value in target.items()}
        for target in targets
    ]


class SetCriterion(nn.Module):
    def __init__(
        self,
        matcher: HungarianMatcher,
        class_loss_weight: float = 1.0,
        keypoint_loss_weight: float = 5.0,
        frames: int = 30,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.class_loss_weight = class_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight
        self.frames = frames

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Sequence[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        matches = [
            self.matcher(outputs, targets, frame)
            for frame in range(self.frames)
        ]
        loss_ce, class_error = self._classification_loss(
            outputs, targets, matches
        )
        loss_kpt, mpjpe = self._keypoint_loss(outputs, targets, matches)
        return {
            "loss_ce": loss_ce,
            "loss_kpt": loss_kpt,
            "class_error": class_error,
            "mpjpe_mm": mpjpe,
            "loss": (
                self.class_loss_weight * loss_ce
                + self.keypoint_loss_weight * loss_kpt
            ),
        }

    def _classification_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Sequence[Dict[str, Tensor]],
        matches: Sequence[Sequence[Match]],
    ) -> tuple[Tensor, Tensor]:
        logits = outputs["pred_logits"]
        target_classes = torch.zeros(
            logits.shape[:3], dtype=torch.long, device=logits.device
        )
        matched_logits = []
        matched_classes = []
        for frame, frame_matches in enumerate(matches):
            for batch_index, (source, destination) in enumerate(frame_matches):
                if source.numel() == 0:
                    continue
                classes = targets[batch_index]["kpt_cls"][frame, destination]
                classes = classes.reshape(-1)
                target_classes[frame, batch_index, source] = classes
                matched_logits.append(logits[frame, batch_index, source])
                matched_classes.append(classes)

        loss_ce = F.cross_entropy(logits.flatten(0, 2), target_classes.flatten())
        if matched_logits:
            predictions = torch.cat(matched_logits).argmax(dim=-1)
            expected = torch.cat(matched_classes)
            class_error = 100.0 * (predictions != expected).float().mean()
        else:
            class_error = logits.new_tensor(0.0)
        return loss_ce, class_error

    def _keypoint_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Sequence[Dict[str, Tensor]],
        matches: Sequence[Sequence[Match]],
    ) -> tuple[Tensor, Tensor]:
        predicted = []
        expected = []
        for frame, frame_matches in enumerate(matches):
            for batch_index, (source, destination) in enumerate(frame_matches):
                if source.numel() == 0:
                    continue
                predicted.append(outputs["pred_kpt"][frame, batch_index, source])
                expected.append(targets[batch_index]["kpt"][frame, destination])

        if not predicted:
            zero = outputs["pred_kpt"].sum() * 0.0
            return zero, zero.detach()
        predicted_tensor = torch.cat(predicted)
        expected_tensor = torch.cat(expected)
        loss_kpt = F.mse_loss(
            predicted_tensor, expected_tensor, reduction="sum"
        ) / predicted_tensor.shape[0]
        mpjpe = mean_per_joint_position_error(
            predicted_tensor, expected_tensor
        )
        return loss_kpt, mpjpe


def mean_per_joint_position_error(
    predicted: Tensor, expected: Tensor
) -> Tensor:
    predicted_joints = predicted.reshape(-1, 21, 3)
    expected_joints = expected.reshape(-1, 21, 3)
    return torch.linalg.vector_norm(
        expected_joints - predicted_joints, dim=-1
    ).mean() * 1000.0


def build_criterion(
    class_cost: float,
    keypoint_cost: float,
    class_loss_weight: float,
    keypoint_loss_weight: float,
    frames: int = 30,
) -> SetCriterion:
    return SetCriterion(
        HungarianMatcher(class_cost, keypoint_cost),
        class_loss_weight,
        keypoint_loss_weight,
        frames,
    )
