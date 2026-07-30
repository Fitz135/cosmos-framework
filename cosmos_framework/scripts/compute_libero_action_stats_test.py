# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_framework.scripts.compute_libero_action_stats import (
    build_stats_document,
    write_stats_document,
)


def _write_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 10.0, "total_episodes": 2, "total_tasks": 1}))
    actions = [
        [0.0, 0.1, -0.1, 0.0, 0.0, 0.0, -1.0],
        [0.2, 0.0, -0.2, 0.1, 0.0, 0.0, 1.0],
        [-0.2, 0.3, 0.0, 0.0, -0.2, 0.0, -1.0],
        [0.1, -0.1, 0.2, 0.0, 0.0, 0.3, 1.0],
    ]
    table = pa.table({"action": pa.array(actions, type=pa.list_(pa.float32(), 7))})
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")


def test_build_stats_document_is_deterministic_and_ten_dimensional(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    first = build_stats_document(tmp_path, dataset_name="test-libero", chunk_length=8)
    second = build_stats_document(tmp_path, dataset_name="test-libero", chunk_length=8)

    assert first == second
    assert first["metadata"]["dataset_fps"] == 10.0
    assert first["metadata"]["num_frames_stats"] == 4
    assert first["metadata"]["chunk_length"] == 8
    assert len(first["metadata"]["source_fingerprint_sha256"]) == 64
    for block in ("global", "global_raw"):
        for key in ("mean", "std", "min", "max", "q01", "q99"):
            assert len(first[block][key]) == 10
    assert first["global"]["mean"][3:9] == [0.0] * 6
    assert first["global"]["std"][3:9] == [1.0] * 6


def test_write_stats_document_round_trips_json(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    document = build_stats_document(tmp_path, dataset_name="test-libero")
    output = tmp_path / "stats.json"

    write_stats_document(document, output)

    assert json.loads(output.read_text()) == document
