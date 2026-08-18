# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Deterministic logical seed contract for LIBERO evaluation requests."""

from __future__ import annotations

import hashlib
import json

from cosmos_framework.utils.rng import ARCH_INVARIANT_RNG_SEED_CONTRACT, validate_arch_invariant_seed

SAMPLING_SEED_CONTRACT = f"sha256-canonical-json-first64-mask63-v1+{ARCH_INVARIANT_RNG_SEED_CONTRACT}"


def _validate_decision_identity(
    base_seed: int,
    task_suite: str,
    task_id: int,
    trial_id: int,
    decision_index: int,
) -> None:
    integer_fields = {
        "base_seed": base_seed,
        "task_id": task_id,
        "trial_id": trial_id,
        "decision_index": decision_index,
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    if not isinstance(task_suite, str) or not task_suite.strip():
        raise ValueError("task_suite must be a non-empty string.")


def stable_decision_seed(
    base_seed: int,
    task_suite: str,
    task_id: int,
    trial_id: int,
    decision_index: int,
) -> int:
    """Map full decision identity to a stable signed-63-bit logical sampling seed."""
    _validate_decision_identity(base_seed, task_suite, task_id, trial_id, decision_index)
    payload = json.dumps(
        {
            "base_seed": base_seed,
            "task_suite": task_suite,
            "task_id": task_id,
            "trial_id": trial_id,
            "decision_index": decision_index,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def stable_episode_seed(base_seed: int, task_suite: str, task_id: int, trial_id: int) -> int:
    """Return the complete, slot-independent uint32 simulator episode seed."""
    # Reuse strict identity validation, while domain-separating simulator
    # randomness from policy-decision randomness.
    _validate_decision_identity(base_seed, task_suite, task_id, trial_id, 0)
    payload = json.dumps(
        {
            "base_seed": base_seed,
            "purpose": "libero_environment",
            "task_suite": task_suite,
            "task_id": task_id,
            "trial_id": trial_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def validate_sampling_seed(value: object, *, name: str = "seed") -> int:
    """Validate a logical sampling seed before it crosses the HTTP boundary."""
    return validate_arch_invariant_seed(value, name=name)
