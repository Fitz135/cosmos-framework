# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import (
    LIBEROLeRobotDataset,
    _resolve_video_keys,
)

_IMAGE = "observation.images.image"
_WRIST = "observation.images.wrist_image"
_IMAGE2 = "observation.images.image2"


def _info(*keys: str) -> dict:
    return {"features": {key: {"dtype": "video"} for key in keys}}


def test_resolve_video_keys_preserves_canonical_default() -> None:
    assert _resolve_video_keys(_info(_IMAGE, _WRIST), "concat_view", _WRIST) == [_IMAGE, _WRIST]


def test_resolve_video_keys_accepts_community_image2() -> None:
    assert _resolve_video_keys(_info(_IMAGE, _IMAGE2), "concat_view", _IMAGE2) == [_IMAGE, _IMAGE2]
    assert _resolve_video_keys(_info(_IMAGE, _IMAGE2), "wrist_image", _IMAGE2) == [_IMAGE2]


def test_resolve_video_keys_rejects_missing_feature() -> None:
    with pytest.raises(ValueError, match="missing video feature"):
        _resolve_video_keys(_info(_IMAGE), "concat_view", _IMAGE2)


def test_eight_action_chunk_uses_nine_observation_frames() -> None:
    dataset = object.__new__(LIBEROLeRobotDataset)
    dataset._chunk_length = 8
    dataset._mode = "wam"
    dataset._camera_mode = "concat_view"
    dataset._valid_cum = np.asarray([1], dtype=np.int64)
    dataset._ep_starts = np.asarray([0], dtype=np.int64)
    dataset._ep_vals = np.asarray([0], dtype=np.int64)
    dataset._episodes = {0: {}}
    dataset._row_timestamp = np.arange(9, dtype=np.float64) / 10.0
    dataset._row_action = np.zeros((9, 7), dtype=np.float32)
    dataset._row_task = np.zeros(9, dtype=np.int64)
    dataset._tasks = {0: "pick up the object"}
    dataset._rotation_space = "6d"
    dataset._load_video = lambda _episode, timestamps: torch.zeros((len(timestamps), 3, 4, 8))
    dataset._build_result = lambda **kwargs: kwargs

    result = dataset._build_item(0)

    assert result["video"].shape == (9, 3, 4, 8)
    assert result["action"].shape == (8, 10)
