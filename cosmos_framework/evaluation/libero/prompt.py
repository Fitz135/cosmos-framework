# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Training-parity prompt construction for LIBERO policy evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import torch

from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter

LIBERO_CONCAT_VIEW_DESCRIPTION = (
    "The left half shows the third-person view; the right half shows the wrist-mounted camera."
)


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _scalar_float(value: object, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar, got tensor shape {tuple(value.shape)}.")
        result = float(value.item())
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite scalar number, got {value!r}.")
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    return result


def _image_size_tensor(image_size: torch.Tensor | Sequence[int]) -> torch.Tensor:
    if isinstance(image_size, torch.Tensor):
        if image_size.ndim != 1 or image_size.numel() < 2:
            raise ValueError(
                "image_size must be a one-dimensional tensor containing at least post-pad height and width, "
                f"got shape {tuple(image_size.shape)}."
            )
        result = image_size.detach().cpu()
    else:
        if isinstance(image_size, (str, bytes)):
            raise ValueError("image_size must contain numeric height and width values.")
        try:
            result = torch.as_tensor(list(image_size))
        except (TypeError, ValueError) as error:
            raise ValueError("image_size must contain numeric height and width values.") from error
        if result.ndim != 1 or result.numel() < 2:
            raise ValueError("image_size must contain at least post-pad height and width.")

    height = _scalar_float(result[0], "image_size[0]")
    width = _scalar_float(result[1], "image_size[1]")
    if height <= 0 or width <= 0 or not height.is_integer() or not width.is_integer():
        raise ValueError(f"image_size height and width must be positive integers, got {(height, width)}.")
    return result


def build_libero_json_prompt(
    task_description: str,
    *,
    video: torch.Tensor,
    image_size: torch.Tensor | Sequence[int],
    conditioning_fps: float | int | torch.Tensor,
    action_chunk_size: int,
    idle_frames: int,
) -> str:
    """Build the exact JSON string used by LIBERO action-policy training.

    ``video`` is the post-padding ``[C, T, H, W]`` tensor and ``image_size`` is
    the post-padding size metadata produced by the action resize transform. The
    task stays in ``actions[].description``; camera layout metadata is supplied
    separately so the shared formatter places it under ``cinematography``.
    """
    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("task_description must be a non-empty string.")
    chunk_size = _validate_positive_int(action_chunk_size, "action_chunk_size")
    if isinstance(idle_frames, bool) or not isinstance(idle_frames, int):
        raise ValueError(f"idle_frames must be an integer, got {idle_frames!r}.")
    if not 0 <= idle_frames <= chunk_size:
        raise ValueError(f"idle_frames must be in [0, {chunk_size}], got {idle_frames}.")
    fps = _scalar_float(conditioning_fps, "conditioning_fps")
    if fps <= 0:
        raise ValueError(f"conditioning_fps must be positive, got {fps}.")

    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        shape = tuple(video.shape) if isinstance(video, torch.Tensor) else None
        raise ValueError(f"video must be a torch tensor with shape [C, T, H, W], got {shape}.")
    if video.shape[0] != 3:
        raise ValueError(f"video must have three RGB channels, got shape {tuple(video.shape)}.")
    expected_frames = chunk_size + 1
    if video.shape[1] != expected_frames:
        raise ValueError(f"video must contain action_chunk_size + 1 = {expected_frames} frames, got {video.shape[1]}.")

    size_tensor = _image_size_tensor(image_size)
    post_pad_height = int(size_tensor[0].item())
    post_pad_width = int(size_tensor[1].item())
    if tuple(video.shape[-2:]) != (post_pad_height, post_pad_width):
        raise ValueError(
            "video spatial shape must match post-pad image_size: "
            f"video={tuple(video.shape[-2:])}, image_size={(post_pad_height, post_pad_width)}."
        )

    data_dict = {
        "ai_caption": task_description.strip(),
        "viewpoint": "concat_view",
        "additional_view_description": LIBERO_CONCAT_VIEW_DESCRIPTION,
        "video": video,
        "image_size": size_tensor,
        "conditioning_fps": torch.tensor(fps, dtype=torch.float32),
        "mode": "wam",
        "action": torch.zeros((chunk_size, 10), dtype=torch.float32),
        "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
    }
    formatted = ActionPromptJsonFormatter()(data_dict)["ai_caption"]
    if not isinstance(formatted, dict):
        raise RuntimeError("ActionPromptJsonFormatter unexpectedly returned a non-dictionary caption.")
    return json.dumps(formatted)
