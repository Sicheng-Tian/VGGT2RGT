# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT Distillation: train a 4-block student VGGT to mimic the last-block features
of the full 24-block teacher using cosine similarity loss.

Key design:
- Teacher: full VGGT (depth=24 in aggregator), frozen.
- Student: VGGT with depth=4 in aggregator; DINO encoder (patch_embed) frozen.
- Feature alignment: extract patch tokens from the LAST block output of each model,
  then apply adaptive average pooling to match spatial resolution.
- Loss: cosine similarity (1 - cos) over patch token features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VGGTDistiller(nn.Module):
    """
    Distillation wrapper that holds a teacher and a student VGGT.

    During training:
        - Teacher is entirely frozen (no gradients).
        - Student's patch_embed (DINO encoder) is frozen; aggregator + heads are trainable.
        - A cosine similarity loss is computed between the last-block patch features
          of student and teacher.

    Args:
        teacher (nn.Module): Pretrained full VGGT, will be set to eval() and no_grad.
        student (nn.Module): Student VGGT (depth=4 aggregator).
        dino_freeze (bool): If True, freeze student's patch_embed. Default: True.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, dino_freeze: bool = True):
        super().__init__()

        self.teacher = teacher
        self.student = student

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        if dino_freeze:
            for param in self.student.aggregator.patch_embed.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        images: Tensor,
        target_size: int | None = None,
    ) -> tuple[Tensor, dict]:
        """
        Compute distillation loss: cosine similarity between student and teacher
        last-block patch features.

        Args:
            images (Tensor): Input images, shape [S, 3, H, W] or [B, S, 3, H, W], range [0, 1].
            target_size (int, optional): Target image size for feature alignment.
                Both student and teacher will pool to (target_size // patch_size)^2 patches.
                If None, uses the native patch grid of each model.

        Returns:
            tuple:
                - loss (Tensor): Scalar distillation loss.
                - metrics (dict): Logging dict with 'cos_sim', 'num_patches', etc.
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)

        B = images.shape[0]

        with torch.no_grad():
            teacher_features = self.teacher.get_last_block_features(
                images, exclude_special_tokens=True, target_size=target_size
            )

        student_features = self.student.get_last_block_features(
            images, exclude_special_tokens=True, target_size=target_size
        )

        teacher_flat = teacher_features.view(B, -1, teacher_features.shape[-1])
        student_flat = student_features.view(B, -1, student_features.shape[-1])

        if teacher_flat.shape[1] != student_flat.shape[1]:
            student_flat = self._align_num_patches(student_flat, teacher_flat.shape[1])

        cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        loss = 1.0 - cos_sim.mean()

        metrics = {
            "distill_loss": loss.detach(),
            "cos_sim": cos_sim.detach(),
            "teacher_patches": teacher_flat.shape[1],
            "student_patches": student_flat.shape[1],
            "teacher_features_shape": tuple(teacher_features.shape),
            "student_features_shape": tuple(student_features.shape),
        }

        return loss, metrics

    def _align_num_patches(self, student_flat: Tensor, target_num_patches: int) -> Tensor:
        """
        Adaptively pool student features to match the target number of patches.

        Args:
            student_flat (Tensor): [B, P_student, C]
            target_num_patches (int): Desired number of patches.

        Returns:
            Tensor: [B, target_num_patches, C]
        """
        B, P_s, C = student_flat.shape
        P_t_sqrt = int(target_num_patches ** 0.5)
        H_s = int(P_s ** 0.5)
        W_s = H_s
        P_t = P_t_sqrt * P_t_sqrt

        x = student_flat.view(B, H_s, W_s, C).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (P_t_sqrt, P_t_sqrt))
        return x.permute(0, 2, 3, 1).reshape(B, P_t, C)


def create_student_vggt(embed_dim: int = 1024, depth: int = 4, **kwargs) -> nn.Module:
    """
    Create a student VGGT with a reduced aggregator depth.

    Uses the same config as the teacher but with fewer attention blocks.
    The patch_embed (DINO encoder) is preserved so it can be frozen separately.

    Args:
        embed_dim (int): Embedding dimension. Default: 1024 (matches VGGT-1B).
        depth (int): Number of blocks in the aggregator. Default: 4.
        **kwargs: Additional arguments passed to VGGT.__init__.

    Returns:
        VGGT: Student model initialised fresh (random weights).
    """
    from vggt.models.vggt import VGGT
    return VGGT(embed_dim=embed_dim, **kwargs)
