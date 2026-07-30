# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compute rot6d action-normalization statistics for a LIBERO LeRobot dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from cosmos_framework.data.generator.action.libero_pose_utils import libero_rotation_format
from cosmos_framework.data.generator.action.pose_utils import convert_rotation

_ACTION_FEATURE = "action"
_ROTATION_DIMS = slice(3, 9)


def _action_parquet_paths(dataset_root: Path) -> list[Path]:
    paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LIBERO parquet files found under {dataset_root / 'data'}.")
    return paths


def load_raw_actions(dataset_root: Path) -> tuple[np.ndarray, list[Path]]:
    """Load the stored 7D ``[dpos, axisangle, gripper]`` actions."""
    paths = _action_parquet_paths(dataset_root)
    parts = [
        np.asarray(pq.read_table(path, columns=[_ACTION_FEATURE])[_ACTION_FEATURE].to_pylist(), dtype=np.float32)
        for path in paths
    ]
    actions = np.concatenate(parts, axis=0)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected LIBERO actions with shape [N, 7], got {actions.shape}.")
    if not np.isfinite(actions).all():
        raise ValueError("LIBERO action parquet contains NaN or Inf values.")
    return actions, paths


def convert_actions_to_rot6d(raw_actions: np.ndarray) -> np.ndarray:
    """Apply the same axis-angle -> rot6d conversion as ``LIBEROLeRobotDataset``."""
    raw = torch.from_numpy(np.ascontiguousarray(raw_actions)).float()
    rotation_matrix = convert_rotation(raw[:, 3:6], input_format="axisangle", output_format="matrix")
    rotation = convert_rotation(
        rotation_matrix,
        input_format="matrix",
        output_format=libero_rotation_format("6d"),
    )
    converted = torch.cat([raw[:, 0:3], rotation, raw[:, 6:7]], dim=-1)
    result = converted.cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (raw_actions.shape[0], 10):
        raise AssertionError(f"Expected converted actions with shape [N, 10], got {result.shape}.")
    return result


def _summarize(actions: np.ndarray) -> dict[str, list[float]]:
    q01, q99 = np.quantile(actions, [0.01, 0.99], axis=0)
    values = {
        "mean": actions.mean(axis=0, dtype=np.float64),
        "std": actions.std(axis=0, dtype=np.float64),
        "min": actions.min(axis=0),
        "max": actions.max(axis=0),
        "q01": q01,
        "q99": q99,
    }
    return {key: [float(value) for value in array] for key, array in values.items()}


def _rotation_passthrough_stats(raw_stats: dict[str, list[float]]) -> dict[str, list[float]]:
    """Build the ``global`` block where rotation dims stay in model space."""
    stats = copy.deepcopy(raw_stats)
    for key, fill in {
        "mean": 0.0,
        "std": 1.0,
        "min": -1.0,
        "max": 1.0,
        "q01": -1.0,
        "q99": 1.0,
    }.items():
        stats[key][_ROTATION_DIMS] = [fill] * 6
    return stats


def _dataset_fingerprint(dataset_root: Path, action_paths: list[Path]) -> str:
    """Hash action parquet plus small metadata files, excluding videos."""
    candidates = [
        dataset_root / "meta" / "info.json",
        dataset_root / "meta" / "tasks.parquet",
        dataset_root / "meta" / "stats.json",
        *action_paths,
    ]
    digest = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            continue
        digest.update(path.relative_to(dataset_root).as_posix().encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_stats_document(
    dataset_root: Path,
    *,
    dataset_name: str,
    chunk_length: int = 8,
) -> dict[str, Any]:
    """Compute the complete normalizer document without writing it."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text())
    raw_actions, action_paths = load_raw_actions(dataset_root)
    actions = convert_actions_to_rot6d(raw_actions)
    global_raw = _summarize(actions)
    return {
        "metadata": {
            "embodiment_type": "libero",
            "pose_convention": "frame_wise_relative",
            "pose_coordinate_frame": "native",
            "rotation_format": "6d",
            "action_dim": 10,
            "skip_rotation_dims": list(range(3, 9)),
            "chunk_length": int(chunk_length),
            "dataset_name": dataset_name,
            "dataset_fps": float(info["fps"]),
            "num_frames_stats": int(actions.shape[0]),
            "total_episodes": int(info.get("total_episodes", 0)),
            "total_tasks": int(info.get("total_tasks", 0)),
            "source_fingerprint_sha256": _dataset_fingerprint(dataset_root, action_paths),
        },
        "global": _rotation_passthrough_stats(global_raw),
        "global_raw": global_raw,
    }


def write_stats_document(document: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", default="libero")
    parser.add_argument("--chunk-length", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_length <= 0 or args.chunk_length % 4 != 0:
        raise ValueError(f"--chunk-length must be a positive multiple of 4, got {args.chunk_length}.")
    document = build_stats_document(
        args.dataset_root,
        dataset_name=args.dataset_name,
        chunk_length=args.chunk_length,
    )
    write_stats_document(document, args.output)
    print(
        f"Wrote {document['metadata']['num_frames_stats']} converted actions "
        f"to {args.output} (fingerprint={document['metadata']['source_fingerprint_sha256']})."
    )


if __name__ == "__main__":
    main()
