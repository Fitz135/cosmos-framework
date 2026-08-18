# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pure seed adapters for architecture-invariant random number generation."""

from __future__ import annotations

UINT32_MAX = 2**32 - 1
SIGNED_63_MAX = 2**63 - 1
ARCH_INVARIANT_RNG_SEED_CONTRACT = "mt19937-uint32-identity-or-le-u32-pair-v1"


def validate_arch_invariant_seed(value: object, *, name: str = "seed") -> int:
    """Return a non-negative signed-63-bit logical seed."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= SIGNED_63_MAX:
        raise ValueError(f"{name} must be an integer in [0, {SIGNED_63_MAX}], got {value!r}.")
    return value


def mt19937_seed_key(seed: int | None) -> int | list[int] | None:
    """Adapt a logical seed to NumPy MT19937 without discarding high bits.

    Existing uint32 seeds stay scalar so their historical RandomState stream is
    byte-identical. Larger signed-63-bit seeds become a little-endian uint32 key.
    """
    if seed is None:
        return None
    value = validate_arch_invariant_seed(seed)
    if value <= UINT32_MAX:
        return value
    return [value & UINT32_MAX, value >> 32]
