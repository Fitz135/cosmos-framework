# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.pose_utils import convert_rotation
from cosmos_framework.evaluation.libero.action import (
    framewise_rot6d_to_libero,
    remap_gripper_pm_one,
    validate_action_chunk,
)

pytestmark = pytest.mark.level(0)


def test_validate_action_chunk_returns_finite_float32_copy() -> None:
    source = np.arange(80, dtype=np.float64).reshape(8, 10)

    validated = validate_action_chunk(source, expected_horizon=8, expected_action_dim=10)

    assert validated.shape == (8, 10)
    assert validated.dtype == np.float32
    assert validated.flags.c_contiguous
    validated[0, 0] = -1
    assert source[0, 0] == 0


@pytest.mark.parametrize(
    ("action", "match"),
    [
        (np.zeros((7, 10)), "shape"),
        (np.zeros((8, 9)), "shape"),
        (np.full((8, 10), np.nan), "finite"),
        (np.full((8, 10), np.inf), "finite"),
    ],
)
def test_validate_action_chunk_rejects_shape_and_nonfinite_values(action: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_action_chunk(action, expected_horizon=8, expected_action_dim=10)


def test_validate_action_chunk_rejects_implicit_or_invalid_contract() -> None:
    action = np.zeros((8, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="positive integer"):
        validate_action_chunk(action, expected_horizon=0, expected_action_dim=10)
    with pytest.raises(ValueError, match="detach"):
        validate_action_chunk(
            torch.zeros((8, 10), requires_grad=True),
            expected_horizon=8,
            expected_action_dim=10,
        )


def test_framewise_rot6d_decodes_each_native_frame_without_translation_transform() -> None:
    axis_angle = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.pi / 2],
            [0.2, -0.3, 0.1],
        ],
        dtype=np.float32,
    )
    rot6d = np.asarray(
        convert_rotation(axis_angle, input_format="axisangle", output_format="rot6d"),
        dtype=np.float32,
    )
    translation = np.array([[1, 2, 3], [-1, -2, -3], [0.1, 0.2, 0.3]], dtype=np.float32)
    gripper = np.array([[-1.0], [1.0], [0.25]], dtype=np.float32)
    edge_action = np.concatenate((translation, rot6d, gripper), axis=-1)

    libero_action = framewise_rot6d_to_libero(edge_action)

    assert libero_action.shape == (3, 7)
    np.testing.assert_allclose(libero_action[:, :3], translation, atol=0, rtol=0)
    np.testing.assert_allclose(libero_action[:, 3:6], axis_angle, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(libero_action[:, -1:], gripper, atol=0, rtol=0)


def test_framewise_rot6d_supports_one_frame_and_rejects_degenerate_basis() -> None:
    action = np.array([1, 2, 3, 1, 0, 0, 0, 1, 0, -1], dtype=np.float32)
    converted = framewise_rot6d_to_libero(action)
    np.testing.assert_allclose(converted, [1, 2, 3, 0, 0, 0, -1], atol=1e-6)

    action[3:9] = 0
    with pytest.raises(ValueError, match="degenerate"):
        framewise_rot6d_to_libero(action)


def test_pm_one_adapter_passes_through_and_clips_only_tolerance_noise() -> None:
    action = np.zeros((3, 7), dtype=np.float32)
    action[:, -1] = [-1.0, 0.25, 1.0 + 5e-7]

    remapped = remap_gripper_pm_one(action)

    np.testing.assert_allclose(remapped[:, -1], [-1.0, 0.25, 1.0], atol=0, rtol=0)
    assert action[-1, -1] > 1.0


@pytest.mark.parametrize("gripper", [-1.01, 1.01, np.nan, np.inf])
def test_pm_one_adapter_rejects_invalid_gripper(gripper: float) -> None:
    action = np.zeros(7, dtype=np.float32)
    action[-1] = gripper
    with pytest.raises(ValueError, match=r"finite|\[-1, 1\]"):
        remap_gripper_pm_one(action)


def test_pm_one_adapter_rejects_wrong_action_dimension() -> None:
    with pytest.raises(ValueError, match="dimension 7"):
        remap_gripper_pm_one(np.zeros((8, 10), dtype=np.float32))
