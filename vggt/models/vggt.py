# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from torch import nn

from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.models.aggregator import Aggregator


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
            if enable_point
            else None
        )
        self.depth_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
            )
            if enable_depth
            else None
        )
        self.track_head = (
            TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
            if enable_track
            else None
        )

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[
                    -1
                ]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = (
                images  # store the images for visualization during inference
            )

        return predictions

    def get_last_block_features(
        self,
        images: torch.Tensor,
        exclude_special_tokens: bool = True,
        target_size: int | None = None,
    ) -> torch.Tensor:
        """
        Extract features from the last attention block of the aggregator.

        In the aggregator, each token sequence has the following structure:
            [CLS/camera_token] + [register_tokens] + [patch_tokens] + [register_tokens_2nd]
            indices:        0          1..N_reg-1         N_reg..N_reg+num_patches-1

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length (number of frames), 3: RGB channels, H: height, W: width
            exclude_special_tokens (bool): If True, excludes the camera token and register tokens,
                returning only the patch token features. If False, returns all tokens in original order.
                Default: True
            target_size (int, optional): If provided, applies adaptive average pooling to the spatial
                dimensions so that the output corresponds to the given image size (e.g. 378 for 378x378).
                The number of output patches becomes (target_size // patch_size)^2.
                If None, outputs the raw patch count (H//patch_size * W//patch_size).
                Default: None

        Returns:
            torch.Tensor: Features from the last block:
                - If exclude_special_tokens=True (default):
                    Shape [B, S, num_patches, 2*embed_dim]
                    where num_patches = (H//patch_size)^2 or (target_size//patch_size)^2.
                    B: batch size, S: number of frames, num_patches: flattened spatial patches,
                    2*embed_dim: concatenated frame+global attention features.
                - If exclude_special_tokens=False:
                    Shape [B, S, P_total, 2*embed_dim]
                    where P_total = 1 + num_register_tokens + num_patch_tokens + num_register_tokens.
                    All tokens (special + patch) in their original sequence order.
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        last_block_features = aggregated_tokens_list[-1]

        if exclude_special_tokens:
            patch_tokens = last_block_features[:, :, patch_start_idx:]
            B, S, P, C = patch_tokens.shape
            H_out = images.shape[2] // self.aggregator.patch_size
            W_out = images.shape[3] // self.aggregator.patch_size

            if target_size is not None:
                H_target = target_size // self.aggregator.patch_size
                W_target = target_size // self.aggregator.patch_size
                num_target_patches = H_target * W_target
                patch_tokens = patch_tokens.view(B, S, H_out, W_out, C)
                patch_tokens = patch_tokens.permute(0, 1, 4, 2, 3)
                patch_tokens = nn.functional.adaptive_avg_pool2d(
                    patch_tokens.view(B * S, C, H_out, W_out), (H_target, W_target)
                )
                patch_tokens = patch_tokens.view(B, S, C, H_target, W_target)
                patch_tokens = patch_tokens.permute(0, 1, 3, 4, 2)
                patch_tokens = patch_tokens.view(B, S, num_target_patches, C)
            else:
                patch_tokens = patch_tokens.view(B, S, H_out * W_out, C)

            return patch_tokens

        return last_block_features
