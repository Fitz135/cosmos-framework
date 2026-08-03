# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for action-policy metadata extraction during Edge export."""

from types import SimpleNamespace

import pytest

from cosmos_framework.scripts._export_model_helpers import (
    build_edge_policy_metadata,
    canonicalize_edge_local_processor,
)


def _dataset(*, chunk_length=8, fps=10, embodiment_type="libero"):
    return {
        "_target_": object,
        "chunk_length": chunk_length,
        "fps": fps,
        "embodiment_type": embodiment_type,
    }


def _defaulted_dataset(*, chunk_length=8, fps=10, embodiment_type="libero"):
    del chunk_length, fps, embodiment_type


def test_build_edge_policy_metadata_from_packing_dataloader():
    training_config = SimpleNamespace(
        dataloader_train={
            "dataloader": {
                "datasets": {
                    "libero_all_10fps": {
                        "ratio": 1,
                        "dataset": _dataset(),
                    }
                }
            }
        }
    )

    assert build_edge_policy_metadata(training_config) == {
        "action_chunk_size": 8,
        "conditioning_fps": 10.0,
        "domain_name": "libero",
    }


def test_build_edge_policy_metadata_reads_callable_defaults():
    training_config = SimpleNamespace(
        dataloader_train={
            "dataloader": {
                "datasets": {
                    "libero_all_10fps": {
                        "dataset": {
                            "_target_": ("cosmos_framework.scripts.export_model_policy_test._defaulted_dataset")
                        }
                    }
                }
            }
        }
    )

    assert build_edge_policy_metadata(training_config) == {
        "action_chunk_size": 8,
        "conditioning_fps": 10.0,
        "domain_name": "libero",
    }


def test_build_edge_policy_metadata_keeps_legacy_layout():
    training_config = SimpleNamespace(
        dataloader_train=SimpleNamespace(
            dataloaders=SimpleNamespace(
                action_data=SimpleNamespace(
                    dataloader=SimpleNamespace(
                        dataset={
                            "list_of_datasets": [
                                {
                                    "dataset": _dataset(
                                        chunk_length=16,
                                        fps=20,
                                        embodiment_type="libero_legacy",
                                    )
                                }
                            ]
                        }
                    )
                )
            )
        )
    )

    assert build_edge_policy_metadata(training_config) == {
        "action_chunk_size": 16,
        "conditioning_fps": 20.0,
        "domain_name": "libero_legacy",
    }


def test_build_edge_policy_metadata_rejects_missing_dataset_hierarchy():
    with pytest.raises(ValueError, match="dataloader_train.dataloader.datasets"):
        build_edge_policy_metadata(SimpleNamespace(dataloader_train={}))


def test_build_edge_policy_metadata_rejects_mixed_policy_values():
    training_config = SimpleNamespace(
        dataloader_train={
            "dataloader": {
                "datasets": {
                    "first": {"dataset": _dataset()},
                    "second": {"dataset": _dataset(chunk_length=4)},
                }
            }
        }
    )

    with pytest.raises(ValueError, match="disagree on `action_chunk_size`"):
        build_edge_policy_metadata(training_config)


def test_canonicalize_edge_local_processor_scrubs_export_host_path(tmp_path):
    model_dict = {
        "config": {
            "vlm_config": {
                "tokenizer": {
                    "repository": None,
                    "revision": None,
                    "subdir": "",
                    "tokenizer_type": str(tmp_path),
                }
            }
        }
    }

    assert canonicalize_edge_local_processor(model_dict)
    assert model_dict["config"]["vlm_config"]["tokenizer"] == {
        "repository": None,
        "revision": None,
        "subdir": "",
        "tokenizer_type": "nvidia/Cosmos3-Edge-Reasoner",
    }
