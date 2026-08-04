# Cosmos3-Edge LIBERO Fine-Tuning Development Record

## Status

- Branch: `feature/cosmos3-edge-libero`
- Base branch: `main`
- Current phase: DEV-0012 container CPU-affinity fix
- Failed 10k rjob: `cosmos3-edge-libero-10k-b128-fa3-r2` (failed before
  process-group initialization; retry pending)
- Full training: `cosmos3-edge-libero-full-b128-2361150` succeeded at
  iteration 5000
- Final DCP: `iter_000005000` (30 GiB on GPFS)
- Final HF/safetensors export: `export-final-iter5000` (7.3 GiB)
- Final real-sample action check: finite `[1, 8, 10]` output

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

The DEV-0005/DEV-0006 effective dry-run confirmed FSDP shard degree 8, max batch 128, gradient
accumulation 2, LR `5e-5`, warmup 500, max iteration 5000, checkpoint interval
500, W&B offline, full dataset split, 10 FPS, action chunk 8, `image2`, and the
dataset-specific normalizer. DEV-0009 changes future periodic saves to every
2000 steps. The trainer's final-save guard means a default 5000-step run writes
checkpoints at 2000, 4000, and 5000.
Checkpoint-aligned qualitative visualization is part of the Edge recipe. The
trainer writes the DCP before invoking `on_training_step_end`, where an
`EveryNDrawSample` callback uses EMA weights to generate one joint WAM rollout
every 2000 optimizer steps. It saves a 10 FPS, three-row local video containing
the prediction, clean-latent VAE reconstruction, and ground truth; offline W&B
receives first/middle/last preview frames. The default uses guidance 1.0 and 8
denoising steps, saves only rank 0, and does not write to object storage. This
keeps the output aligned with periodic checkpoints while bounding the pause.
The final checkpoint at step 5000 is written after the training loop and does
not fire the `EveryN` callback; its visualization remains an explicit post hoc
operation.

The completed `cosmos3-edge-libero-full-b128-2361150` process was launched
before this callback was added and could not hot-load it. It was therefore not
interrupted only to enable visualization. The callback applies to future
resumes or new runs; complete checkpoints can also be visualized post hoc
without changing their weights.

Generic behavior and override syntax are mirrored in
`docs/action_policy_libero_posttrain.md`; PJLAB execution evidence and output
paths remain here.


Training jobs use one private 8×H200 node, 120 CPU cores, 1800000 MiB host
memory, host networking/shared memory, gang start, eight shared RDMA devices,
one Mellanox RDMA device, `brainpp.cn/fuse=1`, and the same GPFS mount:

```bash
source /etc/profile.d/ssh-init.sh
rjob submit \
  --name cosmos3-edge-libero-<mode>-b128 \
  --charged-group llmagent_gpu \
  --private-machine group \
  --image registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_f2adb47 \
  --cpu 120 --gpu 8 --memory 1800000 \
  --share-host-shm=True --host-network=true --gang-start=true \
  --custom-resources \
    rdma/mlnx_shared=8 mellanox.com/mlnx_rdma=1 brainpp.cn/fuse=1 \
  --mount gpfs://gpfs2/intern-pretrain-shared02:/mnt/shared-storage-gpfs2/intern-pretrain-shared02 \
  --set-env \
    COSMOS_LIBERO_MODE=<smoke|resume|full> \
    RESUME_CHECKPOINT=<required-for-resume-and-full> \
    PER_RANK_BATCH=128 GRAD_ACCUM=2 -- \
  bash /mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/.cache/cosmos3_edge_libero_job.sh
```

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
  LeRobot's PyAV backend. DEV-0006 made that backend an explicit, portable
  recipe setting after the CUDA image also proved unable to load TorchCodec's
  required FFmpeg libraries.

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

- `cosmos3-edge-libero-smoke-b128-57317639`: failed before model loading
  because the wrapper called the image's system `torchrun`, whose Python
  environment lacked `omegaconf`. The PJLAB job wrapper now puts the repository
  `.venv/bin` first on `PATH`.
- `cosmos3-edge-libero-smoke-b128-r2-39257879`: reached distributed
  initialization and dataset construction, then failed because TorchCodec
  could not load any supported FFmpeg `libavutil.so`. The dataset now exposes a
  backward-compatible `video_backend`, and the Edge recipe selects `pyav`.
- `cosmos3-edge-libero-smoke-b128-r3-9912282`: succeeded on 8×H200 with
  batch/accumulation `128/2`. It loaded all 549 base-model keys while
  intentionally warm-starting `net_ema`, ran 20 steps without OOM, and wrote
  `iter_000000020`. The first compiled step took 88.65 s; steady-state steps
  took about 19.6 s. Rank losses were about `15.12–15.69` at step 1 and
  `14.35–15.14` at step 20.
- `cosmos3-edge-libero-resume-b128-76675783`: succeeded. It restored the model,
  all 2898 optimizer state items, scheduler, trainer, RNG state, and iteration
  20; then ran steps 21–22 and wrote `iter_000000022`. Step-22 rank losses were
  `14.48–15.52`.
- `cosmos3-edge-libero-full-b128-2361150`: started from `iter_000000022` with
  the validated `128/2` setting and succeeded at iteration 5000. The final
  rank-0 loss was `0.5654`; `iter_000005000` saved in 2.98 s.
  The trainer logged `Done with training` at 2026-07-31 14:56:37 +08:00.
  The complete DCP occupies 30 GiB and contains eight model shards of about
  2.53 GB plus eight optimizer shards of about 1.42 GB, together with trainer,
  scheduler, RNG, and metadata state. The checkpoint is at
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/cosmos3-edge-libero-all-10fps/cosmos3_action_libero/action_sft/action_policy_libero_all_edge_10fps_full/checkpoints/iter_000005000`.
  The offline W&B run is
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/cosmos3-edge-libero-all-10fps/cosmos3_action_libero/action_sft/action_policy_libero_all_edge_10fps_full/wandb/offline-run-20260730_112114-pk7bdv6q`.

- Added `PackingDataLoader` support to Edge export policy metadata extraction
  while retaining the legacy joint-dataloader layout. Resolved training YAML
  can now be inspected without a legacy root `_type`, and the LIBERO wrapper
  exposes its default `embodiment_type` for export.
- `cosmos3-edge-libero-export-check-21245457`: exposed the missing root `_type`
  compatibility issue before reading weights.
- `cosmos3-edge-libero-export-check-r2-21420917`: passed raw-config loading and
  exposed that the resolved wrapper config did not contain a policy domain.
- `cosmos3-edge-libero-export-check-r3-80137598`: succeeded on one H200. Its
  generated `checkpoint.json` contains `action_chunk_size=8`,
  `conditioning_fps=10.0`, and `domain_name=libero`.
- Five focused export-metadata tests and the focused Edge recipe/export suite
  passed. The new resolved-config unit test cannot collect on the development
  pod because its existing `transformer_engine` import requires `libcudart`;
  the real H200 export precheck validates that path end to end.
- `cosmos3-edge-libero-export-smoke-53586371`: completed a provisional
  safetensors export from the resume checkpoint. The 7.3 GiB result contains
  two model shards, the 979 MB Edge vision encoder, processor/tokenizer files,
  export manifest, and correct policy metadata.
- `cosmos3-edge-libero-forward-smoke-32878448`: the real sample loaded, but the
  action service's default guardrails imported OpenCV/RetinaFace and failed on
  the container's missing `libxcb.so.1`. The isolated forward checker now
  disables unrelated guardrails.
- `cosmos3-edge-libero-forward-smoke-r2-12363635`: the exported model loaded,
  then the checker exposed an incorrect normalizer-statistics path. It also
  showed that the export's bundled processor was bypassed because the public
  config retained the PJLAB-local processor path.
- `cosmos3-edge-libero-export-smoke-r2-37086228`: an initial attempt to scrub
  the local processor path before model construction selected the registered
  object-store processor and failed for missing credentials. Canonicalization
  was moved to the final public-config rewrite, after local model construction
  and processor bundling.
- `cosmos3-edge-libero-export-smoke-r3-61518455`: succeeded. The public config
  contains `nvidia/Cosmos3-Edge-Reasoner` rather than a PJLAB path, while the
  export remains self-contained through its bundled processor.
- `cosmos3-edge-libero-forward-smoke-r3-10700008`: succeeded on one H200 with
  real merged-LIBERO sample 0. The prompt was “put the white mug on the left
  plate and put the yellow and white mug on the right plate”; input video shape
  was `[3, 9, 256, 512]`; the offline exported model loaded in 10.98 s and a
  two-denoising-step forward completed in 22.57 s; the finite denormalized
  action output had shape `[1, 8, 10]` and range `[-2.3340, 1.4602]`.
- Added checkpoint-aligned qualitative visualization to the Edge training
  recipe. Every save interval now triggers one EMA WAM sample and stores a
  local 10 FPS prediction/VAE-reconstruction/ground-truth video plus offline
  W&B preview frames. Static configuration tests lock its cadence, rank count,
  sampling settings, and local-only output behavior.
- The first visualization dry-run submission was rejected before job creation
  because the one-GPU request was incompatible with host networking.
  `cosmos3-edge-libero-viz-dryrun-r2-66499044` then succeeded and resolved
  exactly one `libero_rollout` callback with one EMA sample, 8 denoising steps,
  10 FPS, local-only output, and cadence 500.
- `cosmos3-edge-libero-viz-smoke-37737720`,
  `cosmos3-edge-libero-viz-smoke-r2-55564857`, and
  `cosmos3-edge-libero-viz-smoke-fsdp8-67119292` failed in the CUDA preflight
  without loading a checkpoint. A development-pod `uv run` had unexpectedly
  synchronized the shared `.venv` from the verified `torch 2.10.0+cu128` to
  `torch 2.13.0+cu130`; the already-running full job was unaffected because
  its process had loaded the old libraries before that change. New validation
  jobs therefore build an isolated node-local CUDA 12.8 environment without
  modifying or restarting the full job.
- `cosmos3-edge-libero-viz-smoke-fsdp8-r3-83728679` and
  `cosmos3-edge-libero-viz-smoke-fsdp8-r4-75714959` failed before CUDA
  preflight because the H200 node's PyPI proxy timed out while fetching NVRTC
  and CMake respectively. The replacement flow copies non-CUDA packages from
  the existing environment, restores the locked CUDA 12.8 packages from the
  shared uv cache, and installs a staged NVRTC wheel only after verifying its
  SHA256
  `a7756528852ef889772a84c6cd89d41dfa74667e24cca16bb31f8f061e3e9994`.
- `cosmos3-edge-libero-viz-smoke-fsdp8-r5-22977600` passed the CUDA preflight
  and reached 8-rank `torchrun`, but an unanchored temporary rsync exclusion
  also removed `wandb/integration/torch`; it failed before model loading and
  produced no training artifact. Anchoring exclusions to the top-level
  `site-packages` CUDA packages fixed this environment-only issue.
- `cosmos3-edge-libero-viz-smoke-fsdp8-r6-84101056` succeeded end to end on
  8×H200. The preflight reported `torch 2.10.0+cu128`, CUDA 12.8 available,
  and all eight H200s. It restored `iter_000000022`, ran iteration 23 with
  rank-0 loss `15.1887`, saved `iter_000000023` in 6.93 s, switched to EMA,
  and completed the 8-step UniPC sample in 8.30 s.
- The validated local artifact is
  `cosmos3_action_libero/action_sft/action_policy_libero_all_edge_10fps_viz_smoke_fsdp8/EveryNDrawSample/Iter000000023/ema_ReplicateID0000_Sample_Iter000000023.mp4`.
  FFprobe reports H.264, `320×576`, 10 FPS, 9 frames, and 0.9 s duration. Its
  three vertically stacked rows are prediction, VAE reconstruction, and ground
  truth. The reconstruction and ground truth are coherent, while the iteration
  23 prediction is still noise-like, as expected for this deliberately early
  checkpoint; this confirms that the visualization exposes qualitative
  convergence rather than masking it.
- Final training, export, and forward validation are complete.
- `cosmos3-edge-libero-final-export-94756306` used the final run config,
  `iter_000005000`, and the GPFS-local Edge HF vision bundle and succeeded on
  one H200. The output is at
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/cosmos3-edge-libero-all-10fps/export-final-iter5000`.
  The 7.3 GiB export has two model shards of 5,000,054,640 and 1,739,338,512
  bytes, a 978,739,880-byte bundled vision encoder, and bundled processor and
  tokenizer files. `checkpoint.json` records action chunk 8, 10 FPS, domain
  `libero`, and EMA weights. The public model and processor configuration
  replaced the export-host-local processor with
  `nvidia/Cosmos3-Edge-Reasoner`; no PJLAB processor path remains in those
  public fields.
- `cosmos3-edge-libero-final-forward-81163290-6-170ed` was stopped without
  starting because its mistyped GPFS source (`gpfs2-intern-...`) caused the
  storage admission webhook to receive an empty site. The corrected submission
  uses `gpfs://gpfs2/intern-pretrain-shared02:...`.
- `cosmos3-edge-libero-final-forward-r2-4620973-e0067` succeeded on one H200
  with real merged-LIBERO sample 0. The prompt was “put the white mug on the
  left plate and put the yellow and white mug on the right plate”; input video
  shape was `[3, 9, 256, 512]`; model load took 9.903 s and a two-step forward
  took 16.196 s. The denormalized action output was finite with shape
  `[1, 8, 10]` and range `[-1.0180979, 1.0006449]`. The machine-readable report
  is at
  `/mnt/shared-storage-gpfs2/intern-pretrain-shared02/lutianyi/model/cosmos3-edge-libero-all-10fps/forward-final-iter5000.json`.
- `cosmos3-edge-libero-final-validate-62811407-d389a` used the copied
  `pytest` console script, whose absolute shebang still selected the polluted
  shared `.venv`; collection therefore failed on its torch/torchvision version
  mismatch. The corrected validation invokes the node-local Python with
  `-m pytest` so the verified CUDA 12.8 environment is authoritative.
- `cosmos3-edge-libero-final-validate-r2-314760-71180` succeeded on one H200
  with `torch 2.10.0+cu128`: 15 focused recipe, dataset, and export-policy tests
  passed in 3.25 s; the resolved-YAML config test passed with four unrelated
  tests deselected in 5.54 s; and direct TOML parsing passed. Development-pod
  Ruff, format, shell syntax, and diff checks also passed.

### DEV-0009

- Changed both the Python experiment baseline and structured TOML to save
  periodic DCP checkpoints every 2000 steps. The EMA visualization callback now
  uses the same cadence. Trainer final-save behavior preserves step 5000, so a
  default run saves steps 2000, 4000, and 5000 while generating periodic
  rollouts at steps 2000 and 4000.
- Audited Flash Attention without changing its configuration. H200/SM90 orders
  the Cosmos attention backends as `flash3`, `cudnn`, `natten`, then `flash2`,
  and the CUDA 12.8 dependency group declares `flash-attn-3-nv` and
  `flash-attn`. However, the completed-run evidence recorded Torch 2.10/CUDA
  12.8 but not the backend actually selected, so historical Flash Attention 3
  use is not proven.
- The migrated persistent `.venv` currently uses Torch `2.13.0+cu130` and has
  neither `flash-attn-3-nv` nor `flash-attn` installed. A new run that directly
  uses this environment will therefore skip the external Flash Attention 3/2
  backends and select the next compatible backend. Before claiming FA3 for a
  future H200 run, use the CUDA 12.8 dependency group and record both a
  successful `flash_attn_3_nv` import and the selected backend in the job log;
  setting `I4_ATTN_BACKENDS=flash3` provides a fail-fast enforcement check.

### DEV-0010

- The Edge LIBERO launcher now overwrites `I4_ATTN_BACKENDS` with the exact
  allow-list `flash3`; an inherited value cannot restore cuDNN, NATTEN, or FA2
  fallback.
- Added a recipe prelaunch hook and a dedicated CUDA check that verifies SM90,
  imports the ABI-matched `flash-attn-3-nv`, confirms that Cosmos filters the
  backend list to only `flash3`, selects that backend for a representative
  bfloat16 training shape, and executes finite forward and backward kernels.
  Any failure exits before `torchrun`, model loading, or checkpoint I/O.
- The persistent development `.venv` remains Torch 2.13/CUDA 13.0 and is not a
  valid FA3 training environment. GPU validation therefore uses an isolated
  node-local Torch 2.10 environment resolved from the repository's locked CUDA
  dependency group; it does not create a second repository `.venv`.
- H-cluster migration validation established that scheduled jobs must use the
  `evoagi_gpu` charged group and explicitly mount the new GPFS workspace. The
  first two submissions were rejected before job creation by obsolete/default
  charged groups. Preflight `r3` then exposed the missing workspace mount;
  `r4` exposed compute-node package-index timeouts; `r5` showed that offline
  resolution could not discover the cached Torch wheel; `r6` exposed a missing
  NVSHMEM runtime; and `r7` exposed the image's older NCCL symbol set. These
  were environment-only diagnostics and did not start training.
- `cosmos3-edge-libero-fa3-preflight-r8` succeeded on one NVIDIA H200
  (`gpu-lg-cmc-h-h200-0120`) using the locked Torch 2.10.0+cu128 and
  `flash-attn-3-nv` 1.0.3 packages plus their CUDA 12.8 runtime libraries. The
  check constrained Cosmos to `backend=flash3`, selected it for bfloat16
  training tensors, and executed finite forward and backward kernels on SM90.
  The exact terminal evidence is: `Flash Attention 3 preflight passed:
  backend=flash3, package=1.0.3, torch=2.10.0+cu128, cuda=12.8, device=NVIDIA
  H200, sm=90.`
- The persistent evidence log is
  `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework/flash3-preflight/DEV-0010/preflight.log`.
  The validated wheel SHA-256 values are
  `7b4bd23ed63de97456fcc81c26fea9f02ee02ce1112111c4dac0d8cfe574b23e`
  for Torch and
  `ed7b3cf08ffacdeadfaa44ee674a3a5e67e7011829e0e757eb6f3463fb8d443e`
  for Flash Attention 3. The node-local validation environment was ephemeral;
  no second repository `.venv` was retained.

### DEV-0011

- Scheduled a continuation from the migrated 30 GiB `iter_000005000` DCP to a
  total `trainer.max_iter=10000`, retaining batch/accumulation `128/2`, global
  batch 2048, LR `5e-5`, warmup 500, FSDP8, W&B offline, and forced FA3. The
  resume override sets `checkpoint.load_training_state=true`, so model, EMA,
  optimizer, scheduler, trainer, and RNG state continue from iteration 5000.
- All inputs are now available under the migrated GPFS1 workspace: the merged
  LIBERO dataset, Edge DCP/HF processor, Wan VAE, and final iteration-5000
  checkpoint. The new run writes only below
  `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-training/cosmos3-edge-libero-10k-b128-fa3`.
  With `save_iter=2000`, expected checkpoints and EMA rollout videos are at
  iterations 6000, 8000, and 10000.
- The persistent repository `.venv` remains Torch 2.13/CUDA 13.0. A direct
  full sync to CUDA 12.8 was stopped before installation because unpacking the
  CUDA libraries into GPFS was unacceptably slow. The job instead creates one
  ephemeral node-local runtime from the repository's complete locked uv cache,
  while keeping only one persistent repository `.venv`. The runtime uses an
  immutable code export from commit `b97f5840afd9b19c762b5548537e0c30e0f395c0`.
- The launch wrapper is an experiment artifact at
  `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-training/jobs/cosmos3-edge-libero-10k-b128-fa3/launch.sh`
  with SHA-256
  `3aac9cbdf298a1cbb703d1fe4f1ccf425199c8cde82ed78e8f9dfa49014fde14`.
  It validates migrated inputs, constructs the node-local environment, forces
  `I4_ATTN_BACKENDS=flash3`, runs the real FA3 preflight, and only then starts
  eight-rank `torch.distributed.run`.
- `cosmos3-edge-libero-10k-fa3-preflight` failed before environment setup
  because the container lacks `git-lfs`; no training or CUDA kernel ran. The
  wrapper now disables the LFS filter when exporting the pinned code commit.
  `cosmos3-edge-libero-10k-fa3-preflight-r2` then succeeded on H200 node 0119:
  Torch `2.10.0+cu128`, CUDA 12.8, torchvision `0.25.0+cu128`, and
  `flash-attn-3-nv` `1.0.3+cu128.torch210`; the forced FA3 forward/backward
  kernel passed on SM90.
- The first formal submission, `cosmos3-edge-libero-10k-b128-fa3`, requested
  120 CPUs and was stopped while still pending because all 15 matching nodes
  reported insufficient free CPU. `cosmos3-edge-libero-10k-b128-fa3-r2` keeps
  8 H200s, 1,800,000 MiB memory, host networking/shared memory, gang start, and
  RDMA resources while reducing CPU to 96. It was submitted at 2026-08-04
  14:21 +08:00 and remained queued for a matching node at handoff. The durable
  combined bootstrap/training log is `logs/full.log` under the run root above.

### DEV-0012

- `cosmos3-edge-libero-10k-b128-fa3-r2` reached H200 node
  `gpu-lg-cmc-h-h200-0905.host.h.pjlab.org.cn`. Its formal-run preflight passed
  with Torch `2.10.0+cu128`, CUDA 12.8, `flash-attn-3-nv`
  `1.0.3+cu128.torch210`, H200 SM90, forced `backend=flash3`, and finite
  forward/backward kernels.
- All eight ranks then entered `distributed.init()`, but ranks failed at
  `os.sched_setaffinity(0, device.get_cpu_affinity())` with
  `OSError: [Errno 22] Invalid argument`. The failure occurred before process
  group initialization, checkpoint loading, model/data loading, or any
  training iteration and produced no new checkpoint.
- NVML reports CPUs using the host topology, while the rjob container may
  restrict the process to a different Kubernetes cpuset. The old code passed
  the host CPU IDs directly to `sched_setaffinity` and caught only NVML
  exceptions. The fix intersects the NVML set with `os.sched_getaffinity(0)`,
  skips manual binding when the intersection is empty, and treats an OS-level
  rejection as a non-fatal warning.
- Focused regression tests cover a partially overlapping cpuset, disjoint CPU
  sets, and the original `OSError` path. The generic container failure mode and
  behavior are also documented in `docs/faq.md`.
