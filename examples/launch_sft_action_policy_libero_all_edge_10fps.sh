#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Cosmos3-Edge SFT on one merged, 10 FPS LIBERO LeRobot v3 root.
#
# Required:
#   LIBERO_ROOT           merged dataset root containing meta/info.json
# Optional:
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Edge
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   COSMOS3_EDGE_PROCESSOR_PATH
#                         local Edge HF snapshot for offline startup
#   OUTPUT_ROOT           default: outputs/train
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides for smoke/OOM tuning
#
# Default topology: one node, 8 ranks, FSDP8. The recipe uses 128 samples per
# rank and grad_accum=2 (global batch 2048). If memory is insufficient, use:
#   EXTRA_TAIL_OVERRIDES="dataloader_train.max_samples_per_batch=64 trainer.grad_accum_iter=4"
# or:
#   EXTRA_TAIL_OVERRIDES="dataloader_train.max_samples_per_batch=32 trainer.grad_accum_iter=8"

TOML_FILE="examples/toml/sft_config/action_policy_libero_all_edge_10fps.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"

export LIBERO_ROOT="${LIBERO_ROOT:-}"

EXTRA_DATASET_CHECK='
[[ -n "$LIBERO_ROOT" ]] || {
    echo "ERROR: LIBERO_ROOT must point to the merged 10 FPS LIBERO LeRobot root." >&2
    exit 1
}
[[ -f "$LIBERO_ROOT/meta/info.json" ]] || {
    echo "ERROR: missing $LIBERO_ROOT/meta/info.json" >&2
    exit 1
}
grep -q "\"fps\"[[:space:]]*:[[:space:]]*10" "$LIBERO_ROOT/meta/info.json" || {
    echo "ERROR: LIBERO_ROOT metadata must declare fps=10." >&2
    exit 1
}
grep -q "\"observation.images.image\"" "$LIBERO_ROOT/meta/info.json" || {
    echo "ERROR: LIBERO_ROOT must provide observation.images.image." >&2
    exit 1
}
grep -q "\"observation.images.image2\"" "$LIBERO_ROOT/meta/info.json" || {
    echo "ERROR: LIBERO_ROOT must provide observation.images.image2." >&2
    exit 1
}
[[ -f "$WORKDIR/cosmos_framework/data/generator/action/normalizer_stats/libero_10fps_pm_one_native_frame_wise_relative_rot6d.json" ]] || {
    echo "ERROR: dataset-specific action statistics file is missing." >&2
    exit 1
}
if [[ -n "${COSMOS3_EDGE_PROCESSOR_PATH:-}" ]]; then
    [[ -f "$COSMOS3_EDGE_PROCESSOR_PATH/processor_config.json" ]] || {
        echo "ERROR: COSMOS3_EDGE_PROCESSOR_PATH must contain processor_config.json." >&2
        exit 1
    }
fi'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
