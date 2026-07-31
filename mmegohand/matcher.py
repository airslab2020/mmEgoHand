"""Frame-wise Hungarian assignment for hand candidates."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

Match = Tuple[Tensor, Tensor]


class HungarianMatcher(nn.Module):
    def __init__(self, class_cost: float = 5.0, keypoint_cost: float = 25.0):
        super().__init__()
        if class_cost == 0 and keypoint_cost == 0:
            raise ValueError("at least one matching cost must be nonzero")
        self.class_cost = class_cost
        self.keypoint_cost = keypoint_cost

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Sequence[Dict[str, Tensor]],
        frame: int,
    ) -> List[Match]:
        logits = outputs["pred_logits"][frame]
        keypoints = outputs["pred_kpt"][frame]
        matches = []
        for batch_index, target in enumerate(targets):
            target_classes = target["kpt_cls"][frame].reshape(-1)
            target_keypoints = target["kpt"][frame]
            if target_keypoints.shape[0] == 0:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                matches.append((empty, empty))
                continue

            probabilities = logits[batch_index].softmax(dim=-1)
            class_cost = -probabilities[:, target_classes]
            keypoint_cost = torch.cdist(
                keypoints[batch_index], target_keypoints, p=1
            )
            total_cost = (
                self.class_cost * class_cost
                + self.keypoint_cost * keypoint_cost
            )
            source, destination = linear_sum_assignment(total_cost.cpu())
            matches.append(
                (
                    torch.as_tensor(source, dtype=torch.long, device=logits.device),
                    torch.as_tensor(
                        destination, dtype=torch.long, device=logits.device
                    ),
                )
            )
        return matches
