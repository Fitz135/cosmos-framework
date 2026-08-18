# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict action adapters for Cosmos Edge LIBERO evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.libero_pose_utils import libero_rotation_format
from cosmos_framework.data.generator.action.pose_utils import convert_rotation

EDGE_LIBERO_ACTION_DIM = 10
LIBERO_ENV_ACTION_DIM = 7
GRIPPER_ADAPTER_CONTRACT = "pm-one-finite-clamp-v1"


@dataclass(frozen=True)
class PmOneGripperAdapterResult:
    """Adapted action and audit data for one finite ``pm_one`` clamp."""

    action: np.ndarray
    raw_min: float
    raw_max: float
    clipped_count: int
    value_count: int


def _as_float32_array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError(f"{name} is non-differentiable; detach the tensor explicitly before conversion.")
        if value.dtype == torch.bool or value.is_complex():
            raise ValueError(f"{name} must contain real numeric values, got dtype {value.dtype}.")
        raw = value.cpu().to(dtype=torch.float32).numpy()
    else:
        raw = np.asarray(value)

    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values, got dtype {raw.dtype}.")
    array = raw.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def validate_action_chunk(
    action_chunk: Any,
    *,
    expected_horizon: int,
    expected_action_dim: int,
) -> np.ndarray:
    """Validate an Edge policy output and return a contiguous float32 copy."""
    horizon = _positive_int(expected_horizon, "expected_horizon")
    action_dim = _positive_int(expected_action_dim, "expected_action_dim")
    array = _as_float32_array(action_chunk, "action_chunk")
    expected_shape = (horizon, action_dim)
    if array.ndim != 2 or array.shape != expected_shape:
        raise ValueError(f"action_chunk must have shape {expected_shape}, got {array.shape}.")
    return array.copy()


def framewise_rot6d_to_libero(action: Any) -> np.ndarray:
    """Decode native-frame rot6d actions into LIBERO's axis-angle commands.

    Input rows are ``[dxyz(3), rot6d(6), gripper(1)]`` and output rows are
    ``[dxyz(3), axisangle(3), gripper(1)]``. Translation is already a per-step
    delta in LIBERO's native controller frame, so no anchor or frame transform
    is applied. Any leading dimensions are preserved.
    """
    array = _as_float32_array(action, "action")
    if array.ndim < 1 or array.size == 0 or array.shape[-1] != EDGE_LIBERO_ACTION_DIM:
        raise ValueError(
            f"action must have trailing Edge LIBERO dimension {EDGE_LIBERO_ACTION_DIM}, got {array.shape}."
        )

    rotation_6d = array[..., 3:9]
    col0 = rotation_6d[..., :3]
    col1 = rotation_6d[..., 3:]
    min_basis_norm = 1e-6
    degenerate = (
        (np.linalg.norm(col0, axis=-1) <= min_basis_norm)
        | (np.linalg.norm(col1, axis=-1) <= min_basis_norm)
        | (np.linalg.norm(np.cross(col0, col1, axis=-1), axis=-1) <= min_basis_norm)
    )
    if np.any(degenerate):
        raise ValueError("action contains a degenerate rot6d basis.")

    axis_angle = np.asarray(
        convert_rotation(
            rotation_6d,
            input_format=libero_rotation_format("6d"),
            output_format="axisangle",
            normalize_matrix=True,
        ),
        dtype=np.float32,
    )
    converted = np.concatenate((array[..., :3], axis_angle, array[..., 9:10]), axis=-1)
    return np.ascontiguousarray(converted, dtype=np.float32)


def adapt_gripper_pm_one(action: Any) -> PmOneGripperAdapterResult:
    """Apply the versioned finite-clamp ``pm_one`` simulator contract.

    All finite values retain their continuous semantics inside ``[-1, 1]``;
    only values outside that interval are clipped. Shape and finiteness remain
    strict so clipping cannot conceal a malformed policy response.
    """
    array = _as_float32_array(action, "action")
    if array.ndim < 1 or array.size == 0 or array.shape[-1] != LIBERO_ENV_ACTION_DIM:
        raise ValueError(
            f"action must have trailing LIBERO environment dimension {LIBERO_ENV_ACTION_DIM}, got {array.shape}."
        )
    gripper = array[..., -1]
    outside = (gripper < -1.0) | (gripper > 1.0)
    result = array.copy()
    result[..., -1] = np.clip(gripper, -1.0, 1.0)
    return PmOneGripperAdapterResult(
        action=result,
        raw_min=float(np.min(gripper)),
        raw_max=float(np.max(gripper)),
        clipped_count=int(np.count_nonzero(outside)),
        value_count=int(gripper.size),
    )


def remap_gripper_pm_one(action: Any) -> np.ndarray:
    """Return only the adapted action for callers that do not need audit data."""
    return adapt_gripper_pm_one(action).action
