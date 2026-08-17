# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json

import pytest
import torch

from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.evaluation.libero.prompt import (
    LIBERO_CONCAT_VIEW_DESCRIPTION,
    build_libero_json_prompt,
)

pytestmark = pytest.mark.level(0)


def _video() -> torch.Tensor:
    return torch.zeros((3, 9, 256, 512), dtype=torch.uint8)


def test_json_prompt_is_byte_identical_to_training_formatter() -> None:
    video = _video()
    image_size = torch.tensor([256, 512, 256, 512], dtype=torch.long)
    expected_data = {
        "ai_caption": "Pick up the red mug",
        "viewpoint": "concat_view",
        "additional_view_description": LIBERO_CONCAT_VIEW_DESCRIPTION,
        "video": video,
        "image_size": image_size,
        "conditioning_fps": torch.tensor(20.0, dtype=torch.float32),
        "mode": "wam",
        "action": torch.zeros((8, 10), dtype=torch.float32),
        "idle_frames": torch.tensor(0, dtype=torch.long),
    }
    expected_prompt = json.dumps(ActionPromptJsonFormatter()(expected_data)["ai_caption"])

    actual_prompt = build_libero_json_prompt(
        "Pick up the red mug",
        video=video,
        image_size=image_size,
        conditioning_fps=20,
        action_chunk_size=8,
        idle_frames=0,
    )

    assert actual_prompt == expected_prompt


def test_json_prompt_keeps_task_and_camera_layout_in_training_fields() -> None:
    prompt = build_libero_json_prompt(
        "Pick up the red mug",
        video=_video(),
        image_size=[256, 512, 256, 512],
        conditioning_fps=20.0,
        action_chunk_size=8,
        idle_frames=0,
    )

    parsed = json.loads(prompt)
    assert parsed["actions"] == [
        {
            "time": "0:00-0:00",
            "description": "Pick up the red mug.",
            "idle_frame": "0 out of 8.",
        }
    ]
    assert parsed["cinematography"]["framing"] == (
        f"This video contains concatenated views from multiple camera perspectives. {LIBERO_CONCAT_VIEW_DESCRIPTION}"
    )
    assert parsed["fps"] == 20.0
    assert parsed["resolution"] == {"H": 256, "W": 512}
    assert parsed["aspect_ratio"] == "2,1"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"video": torch.zeros((3, 8, 256, 512))}, r"action_chunk_size \+ 1"),
        ({"image_size": [256, 256]}, "spatial shape"),
        ({"conditioning_fps": 0}, "positive"),
        ({"idle_frames": 9}, "idle_frames"),
    ],
)
def test_json_prompt_rejects_inconsistent_runtime_metadata(overrides: dict[str, object], match: str) -> None:
    kwargs: dict[str, object] = {
        "video": _video(),
        "image_size": [256, 512, 256, 512],
        "conditioning_fps": 20,
        "action_chunk_size": 8,
        "idle_frames": 0,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=match):
        build_libero_json_prompt("Pick up the red mug", **kwargs)  # type: ignore[arg-type]


def test_json_prompt_rejects_empty_task() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_libero_json_prompt(
            "   ",
            video=_video(),
            image_size=[256, 512],
            conditioning_fps=20,
            action_chunk_size=8,
            idle_frames=0,
        )
