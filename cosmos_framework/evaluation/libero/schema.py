# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Versioned, strict schemas for a LIBERO checkpoint evaluation profile."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CheckpointFormat(str, Enum):
    """Supported on-disk checkpoint formats."""

    DCP = "dcp"
    HF = "hf"


class CheckpointRole(str, Enum):
    """Whether weights are task-adapted or an explicit zero-shot baseline."""

    BASE = "base"
    FINETUNED = "finetuned"


class WeightsVariant(str, Enum):
    """The single set of parameters selected for policy evaluation."""

    EMA = "ema"
    REGULAR = "regular"


class _StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LiberoCheckpointProfileSpec(_StrictSchema):
    """Partial explicit profile merged with checkpoint-native metadata.

    Every field is optional here because HF policy metadata and DCP resolved
    configs contribute different subsets. Resolution produces the fully
    required LiberoCheckpointProfile below.
    """

    schema_version: Literal[1] = 1
    checkpoint_role: CheckpointRole | None = None
    zero_shot: bool | None = None
    domain_name: Literal["libero"] | None = None
    action_chunk_size: int | None = Field(default=None, gt=0)
    conditioning_fps: float | None = Field(default=None, gt=0)
    effective_action_dim: int | None = Field(default=None, gt=0)
    action_space: Literal["frame_wise_relative", "relative"] | None = None
    rotation_space: Literal["3d", "6d"] | None = None
    pose_coordinate_frame: Literal["native", "world"] | None = None
    action_normalization: Literal["meanstd", "minmax", "quantile", "quantile_rot"] | None = None
    action_stats_path: str | None = None
    action_stats_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    format_prompt_as_json: bool | None = None
    cameras: tuple[str, ...] | None = None
    image_size: int | None = Field(default=None, gt=0)
    rotate_images: bool | None = None
    control_mode: str | None = None
    control_frequency: int | None = Field(default=None, gt=0)
    gripper_mode: str | None = None
    weights_variant: WeightsVariant | None = None
    profile_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("action_stats_path", "control_mode", "gripper_mode")
    @classmethod
    def _nonempty_optional_string(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("cameras")
    @classmethod
    def _valid_optional_cameras(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        normalized = tuple(camera.strip() for camera in value)
        if not normalized or any(not camera for camera in normalized):
            raise ValueError("cameras must contain at least one non-empty camera name")
        if len(set(normalized)) != len(normalized):
            raise ValueError("cameras must be unique and ordered")
        return normalized


class LiberoTargetAdapterSpec(LiberoCheckpointProfileSpec):
    """Explicit simulator-side contract used to adapt weights to LIBERO."""

    adapter_id: str = Field(min_length=1)
    target: Literal["libero"] = "libero"


class LiberoCheckpointProfile(_StrictSchema):
    """Fully resolved policy contract required before starting LIBERO eval."""

    schema_version: Literal[1] = 1
    checkpoint_path: str = Field(min_length=1)
    checkpoint_format: CheckpointFormat
    checkpoint_role: CheckpointRole
    zero_shot: bool
    target_adapter_id: str | None = None
    domain_name: Literal["libero"]
    action_chunk_size: int = Field(gt=0)
    conditioning_fps: float = Field(gt=0)
    effective_action_dim: int = Field(gt=0)
    action_space: Literal["frame_wise_relative", "relative"]
    rotation_space: Literal["3d", "6d"]
    pose_coordinate_frame: Literal["native", "world"]
    action_normalization: Literal["meanstd", "minmax", "quantile", "quantile_rot"]
    action_stats_path: str = Field(min_length=1)
    action_stats_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    format_prompt_as_json: bool
    cameras: tuple[str, ...]
    image_size: int = Field(gt=0)
    rotate_images: bool
    control_mode: str = Field(min_length=1)
    control_frequency: int = Field(gt=0)
    gripper_mode: str = Field(min_length=1)
    weights_variant: WeightsVariant
    profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_sources: tuple[str, ...]

    @field_validator("cameras")
    @classmethod
    def _valid_cameras(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(camera.strip() for camera in value)
        if not normalized or any(not camera for camera in normalized):
            raise ValueError("cameras must contain at least one non-empty camera name")
        if len(set(normalized)) != len(normalized):
            raise ValueError("cameras must be unique and ordered")
        return normalized

    @model_validator(mode="after")
    def _validate_policy_contract(self) -> "LiberoCheckpointProfile":
        if self.checkpoint_role == CheckpointRole.BASE:
            if not self.zero_shot:
                raise ValueError("base checkpoints must be marked zero_shot=true")
            if self.target_adapter_id is None:
                raise ValueError("base checkpoints require an explicit target adapter")
        elif self.zero_shot:
            raise ValueError("finetuned checkpoints must not be marked zero-shot")

        expected_dim = 10 if self.rotation_space == "6d" else 7
        if self.effective_action_dim != expected_dim:
            raise ValueError(
                f"rotation_space={self.rotation_space!r} requires effective_action_dim={expected_dim}, "
                f"got {self.effective_action_dim}"
            )
        return self
