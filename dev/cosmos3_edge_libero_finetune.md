# Cosmos3-Edge LIBERO Fine-Tuning Development Record

## Status

- Branch: `feature/cosmos3-edge-libero`
- Base branch: `main`
- Current phase: DEV-0005 validated and ready to commit
- Full training: not started

## Objective

Fine-tune Cosmos3-Edge as an action policy on the merged LIBERO LeRobot
dataset available on the PJLAB H cluster. The implementation remains separate
from the existing Cosmos3-Nano LIBERO recipes and uses the Edge model's
pretrained action heads.

## Development boundaries

- Generic, reusable user instructions remain in the repository's existing
  `docs/` and `examples/` locations.
- Cluster paths, rjob commands, implementation decisions, experiment output,
  failures, and measurements are maintained in this document.
- Dataset, model weights, W&B artifacts, and rjob logs are not committed.

## Dataset

- Root:
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/data/lerobot-libero`
- LeRobot version: v3.0
- Episodes: 1693
- Frames: 273465
- Tasks: 40
- Native FPS: 10
- External camera: `observation.images.image`
- Wrist camera: `observation.images.image2`
- Stored action: 7D `[dpos(3), axisangle(3), gripper(1)]`
- Gripper convention: approximately `[-1, 1]` (`pm_one`)

The two video features were visually checked from matching frames. `image`
contains the third-person view and `image2` contains the wrist-mounted view.

## Action chunk decision

The official 20 FPS LIBERO recipe uses 16 actions, covering 0.8 seconds. This
dataset is 10 FPS, so the Edge recipe uses 8 actions and 9 observation frames
to preserve the same 0.8-second physical prediction horizon. Eight also
satisfies the loader requirement that chunk length be divisible by four.

## Action normalization

The existing bundled LIBERO statistics were generated for a 20 FPS conversion
with a different gripper convention and must not be reused. DEV-0004 adds a
deterministic statistics CLI that:

1. reads every action parquet in sorted order;
2. converts axis-angle rotation to the same rot6d representation used by the
   training dataset;
3. computes mean, standard deviation, min, max, q01, and q99 over the resulting
   10D actions; and
4. fingerprints the action parquet and small metadata files with SHA-256.

The generated statistics will be committed as a small reproducibility asset;
the source dataset remains outside Git.
The 273465-frame output fingerprint is
`4a64eacab21f27ed949dce5a024d028fe38101b2a69ee2a6ae25eac11a4e84a1`.

## Edge recipe

- Experiment: `action_policy_libero_all_edge_10fps`
- Model base: `EDGE_MODEL_CONFIG`
- Action chunk: 8
- Native FPS: 10
- Camera layout: external view left, wrist view right
- Parallelism: single-node FSDP, shard degree 8
- Target global batch: 2048
- Optimizer LR: `5e-5`, with 5x action-module multipliers
- Training length: 5000 iterations
- Checkpoint interval: 500 iterations
- Tracking: local logs plus offline W&B

The recipe loads the renewed Cosmos3-Edge action-head weights. It skips only
`net_ema` during base initialization and fails if the expected action-head keys
are absent.

The experiment keeps the public `EDGE_MODEL_CONFIG` immutable by deep-copying
it before recipe-specific changes. It disables secondary diffusion-expert
weight initialization because the DCP is the single source of model weights,
retains full activation checkpointing and the 45056-token limit, and trains
`k_norm_und_for_gen` alongside the generation and action modules. It does not
change the Nano recipes, the shared Edge baseline, the inference server, or
closed-loop evaluation.

The paired TOML selects single-node FSDP8, bf16, batch 128 per rank,
gradient accumulation 2, LR `5e-5`, warmup 500, 5000 iterations, checkpoint
interval 500, and W&B offline mode. The launcher validates the merged dataset's
10 FPS metadata, `image2` camera feature, action statistics, base DCP, and Wan
VAE before starting. OOM fallbacks preserve global batch 2048 with `64/4` and
then `32/8`.

## Cluster execution

Every cluster session must source the shared `.bashrc` and enable `proxy_on`.
GPU validation and training require an rjob allocation; no GPU workload runs
on the development pod.

Persistent resources:

- Edge DCP:
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/Cosmos3-Edge-dcp`
- Wan VAE:
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/wan22_vae/Wan2.2_VAE.pth`
- Training root:
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/cosmos3-edge-libero-all-10fps`

Preparation used the image
`registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_f2adb47`,
the `llmagent_gpu` charged group, a private group machine, one H200, 32 CPU
cores, 131072 MiB memory, `brainpp.cn/fuse=1`, and the
`gpfs://gpfs2/intern-pretrain-shared02` mount. The reusable command shape was:

```bash
source /etc/profile.d/ssh-init.sh
rjob submit \
  --name cosmos3-edge-libero-prepare \
  --charged-group llmagent_gpu \
  --private-machine group \
  --image registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_f2adb47 \
  --cpu 32 --gpu 1 --memory 131072 \
  --custom-resources brainpp.cn/fuse=1 \
  --mount gpfs://gpfs2/intern-pretrain-shared02:/mnt/shared-storage-gpfs2/intern-pretrain-shared02 \
  --set-env COSMOS_LIBERO_MODE=prepare -- \
  bash /mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/.cache/cosmos3_edge_libero_job.sh
```

Preparation job history:

- `cosmos3-edge-libero-prepare-56070503`: failed because the allocated H200
  node could not download the VAE through the proxy.
- `cosmos3-edge-libero-prepare-r2-1726033`: failed because a local HF
  checkpoint without the registered Edge YAML selected the wrong root config.
- `cosmos3-edge-libero-prepare-r3-27977290`: failed because the registered
  config still tried to download the Edge processor.
- `cosmos3-edge-libero-prepare-r4-77213900`: stopped after the local processor
  fix exposed an attempted Wan2.2 VAE Hub download.
- `cosmos3-edge-libero-prepare-r5-96028663`: verified the local VAE path was
  injected, then failed because the inherited bucket name still selected the
  object-store backend and required `credentials/gcp_training.secret`.
- `cosmos3-edge-libero-prepare-r6-630658`: succeeded. It ran on an H200 with
  PyTorch `2.10.0+cu128`, loaded both processor and VAE from GPFS, wrote the
  DCP checkpoint, and completed the structured-TOML training dry-run.
- `cosmos3-edge-libero-dryrun-r7-95807184`: rjob returned success, but the log
  exposed an OmegaConf `ConfigKeyError` because the first local-processor
  implementation removed structured tokenizer fields. The result was rejected
  as validation evidence and the recipe was corrected to preserve those fields
  with `None` values.
- `cosmos3-edge-libero-dryrun-r8-63642809`: succeeded without serialization
  errors. The resolved model and dataset tokenizer configs both used the local
  Edge HF snapshot with `repository=None`.

The converter now supports `--vae-path` and redirects compatible
Cosmos3-Edge snapshots to their bundled processor. Supplying a local VAE also
clears the object-store bucket and credential fields, so conversion requires no
Hub or internal object-store access after the artifacts are staged.
The training recipe accepts `COSMOS3_EDGE_PROCESSOR_PATH`; PJLAB jobs set it to
the staged Edge HF snapshot so model and dataset processor construction also
remain offline.

The effective dry-run confirmed FSDP shard degree 8, max batch 128, gradient
accumulation 2, LR `5e-5`, warmup 500, max iteration 5000, checkpoint interval
500, W&B offline, full dataset split, 10 FPS, action chunk 8, `image2`, and the
dataset-specific normalizer.

## Implementation log

### DEV-0004

- Parameterize the wrist camera feature while retaining the NVIDIA default.
- Add deterministic 7D axis-angle to 10D rot6d statistics generation.
- Add unit tests for camera resolution, the 9-frame/8-action window, and
  statistics determinism.
- Generate and validate the dataset-specific statistics file.
- Validation: six focused tests passed; Ruff and format checks passed.
- Real-data validation loaded all 1693 episodes and produced a finite sample
  with video shape `[3, 9, 256, 512]` and action shape `[8, 10]`.
- The GPU-less development pod's default TorchCodec path cannot load
  `libnppicc.so.12`; the CPU integration check therefore explicitly selected
  LeRobot's PyAV backend. The rjob smoke test must verify the default TorchCodec
  path in the CUDA container before full training.

### DEV-0005

- Added and registered `action_policy_libero_all_edge_10fps`.
- Added its paired structured TOML and launcher.
- Added configuration tests covering Edge model semantics, pretrained action
  head loading, 10 FPS/8-action/full-dataset behavior, `image2`, statistics,
  FSDP8, global batch 2048, offline W&B, and save cadence.
- Updated `docs/action_policy_libero_posttrain.md`, `docs/training.md`, and
  `examples/README.md` with only reusable behavior and launch instructions.
  Cluster-specific paths and execution evidence remain in this document.
- Extended HF-to-DCP conversion with an optional local VAE path and
  checkpoint-bundled Edge processor selection; added five focused regression
  tests for the offline redirects.
- Added optional local Edge processor selection to the recipe and launcher so
  actual model construction does not require Hub access.
- Validation: Ruff, format, shell syntax, diff, focused recipe/converter tests,
  all example TOML schema tests, launcher mock validation against the real
  dataset, and the corrected H200 dry-run passed. The full TOML loader suite
  has two pre-existing failures on the GPU-less development pod because
  `transformer_engine` requires CUDA runtime discovery; the shipped-example
  schema subset passed.

### DEV-0006

Pending: record rjob smoke training, resume validation, full training, export,
and real-sample forward validation.
