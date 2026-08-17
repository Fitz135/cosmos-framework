# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import numpy as np
import pytest

from cosmos_framework.data.generator.action.transforms import build_sequence_plan_from_mode
from cosmos_framework.evaluation.libero.observation import (
    EDGE_LIBERO_CAMERAS,
    EDGE_LIBERO_IMAGE_SIZE,
    build_libero_concat_frame,
    build_wam_conditioning_video,
)

pytestmark = pytest.mark.level(0)


def _observation(agentview: np.ndarray, wrist: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "agentview_image": agentview,
        "robot0_eye_in_hand_image": wrist,
    }


def _build(observation: dict[str, np.ndarray], **overrides: object) -> np.ndarray:
    kwargs = {
        "cameras": EDGE_LIBERO_CAMERAS,
        "image_size": EDGE_LIBERO_IMAGE_SIZE,
        "rotate_180": True,
        "flip_images": False,
    }
    kwargs.update(overrides)
    return build_libero_concat_frame(observation, **kwargs)  # type: ignore[arg-type]


def test_concat_frame_uses_fixed_camera_order_and_rotates_each_view() -> None:
    agentview = np.full((256, 256, 3), [10, 20, 30], dtype=np.uint8)
    wrist = np.full((256, 256, 3), [40, 50, 60], dtype=np.uint8)
    agentview[-1, -1] = [1, 2, 3]
    wrist[-1, -1] = [4, 5, 6]

    frame = _build(_observation(agentview, wrist))

    assert frame.shape == (256, 512, 3)
    assert frame.dtype == np.uint8
    np.testing.assert_array_equal(frame[0, 0], [1, 2, 3])
    np.testing.assert_array_equal(frame[0, 256], [4, 5, 6])
    np.testing.assert_array_equal(frame[-1, -1], [40, 50, 60])


def test_concat_frame_resizes_views_independently_and_scales_float_pixels() -> None:
    agentview = np.full((4, 6, 3), [1.0, 0.5, 0.0], dtype=np.float32)
    wrist = np.full((8, 3, 3), [0.0, 0.25, 1.0], dtype=np.float32)

    frame = _build(_observation(agentview, wrist))

    assert frame.shape == (256, 512, 3)
    np.testing.assert_array_equal(frame[128, 128], [255, 128, 0])
    np.testing.assert_array_equal(frame[128, 384], [0, 64, 255])


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"cameras": ("wrist", "agentview")}, "camera order"),
        ({"image_size": 224}, "image_size=256"),
        ({"rotate_180": False}, "rotate_180=True"),
    ],
)
def test_concat_frame_rejects_non_edge_observation_contract(override: dict[str, object], match: str) -> None:
    observation = _observation(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match=match):
        _build(observation, **override)


def test_concat_frame_rejects_missing_or_invalid_camera_images() -> None:
    valid = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="robot0_eye_in_hand_image"):
        _build({"agentview_image": valid})

    nonfinite = np.zeros((2, 2, 3), dtype=np.float32)
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _build(_observation(nonfinite, valid))


def test_wam_video_repeats_only_the_current_frame_as_shape_template() -> None:
    frame = np.arange(256 * 512 * 3, dtype=np.uint32).reshape(256, 512, 3).astype(np.uint8)

    video = build_wam_conditioning_video(frame, action_chunk_size=8)

    assert video.shape == (3, 9, 256, 512)
    assert video.dtype == np.uint8
    assert video.flags.c_contiguous
    for frame_index in range(video.shape[1]):
        np.testing.assert_array_equal(video[:, frame_index], frame.transpose(2, 0, 1))

    sequence_plan = build_sequence_plan_from_mode(
        mode="wam",
        video_length=video.shape[1],
        action_length=8,
        has_text=True,
    )
    assert sequence_plan.condition_frame_indexes_vision == [0]
    assert sequence_plan.condition_frame_indexes_action == []


def test_wam_video_rejects_wrong_frame_shape_dtype_or_chunk_size() -> None:
    frame = np.zeros((256, 512, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="positive integer"):
        build_wam_conditioning_video(frame, action_chunk_size=0)
    with pytest.raises(ValueError, match="shape"):
        build_wam_conditioning_video(frame[:, :-1], action_chunk_size=8)
    with pytest.raises(ValueError, match="dtype uint8"):
        build_wam_conditioning_video(frame.astype(np.float32), action_chunk_size=8)
