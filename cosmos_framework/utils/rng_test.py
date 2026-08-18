# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmos_framework.utils.misc import arch_invariant_rand
from cosmos_framework.utils.rng import SIGNED_63_MAX, mt19937_seed_key

pytestmark = pytest.mark.level(0)


def test_uint32_seed_keeps_legacy_random_state_stream() -> None:
    key = mt19937_seed_key(7)

    assert key == 7
    np.testing.assert_array_equal(
        np.random.RandomState(key).standard_normal(16),
        np.random.RandomState(7).standard_normal(16),
    )


def test_v6_signed63_seed_uses_both_uint32_words_deterministically() -> None:
    seed = 7221137112376841976
    key = mt19937_seed_key(seed)

    assert key == [3382482680, 1681302001]
    first = np.random.RandomState(key).standard_normal(16)
    second = np.random.RandomState(mt19937_seed_key(seed)).standard_normal(16)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, np.random.RandomState(key[0]).standard_normal(16))


def test_arch_invariant_rand_accepts_v6_seed_at_real_rng_boundary() -> None:
    seed = 7221137112376841976
    expected = np.random.RandomState([3382482680, 1681302001]).standard_normal(16).astype(np.float32)

    actual = arch_invariant_rand((16,), torch.float32, "cpu", seed=seed)

    np.testing.assert_array_equal(actual.numpy(), expected)


def test_high_seed_word_changes_random_state_stream() -> None:
    low_word = 7
    low_only = np.random.RandomState(mt19937_seed_key(low_word)).standard_normal(16)
    with_high_word = np.random.RandomState(mt19937_seed_key((1 << 32) + low_word)).standard_normal(16)

    assert not np.array_equal(low_only, with_high_word)


@pytest.mark.parametrize("seed", [True, -1, SIGNED_63_MAX + 1])
def test_invalid_logical_seed_is_rejected(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        mt19937_seed_key(seed)  # type: ignore[arg-type]
