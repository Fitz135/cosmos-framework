# AGENTS.md — Cosmos-Framework

Read this file first — it is the canonical map for navigating the Cosmos repository and stays up to date.

**Cosmos** is a framework for training and serving world foundation models. Everything lives in a single top-level `cosmos_framework/` Python package:

- **Training infrastructure** — top-level subpackages under `cosmos_framework/` (data, model, trainer, callbacks, checkpoint, …).
- **Inference infrastructure** — `cosmos_framework/inference/` (Diffusers / Transformers / vLLM-friendly inference core, online serving via Ray + Gradio).
- **Backend packages** — `packages/{diffusers,transformers,vllm}-cosmos3/` provide library-style shims that load Cosmos3 checkpoints into the respective ecosystems.
- **Entry-point scripts** — `cosmos_framework/scripts/` (`train.py`, `inference.py`, `export_model.py`, …) invoked as `python -m cosmos_framework.scripts.<name>`. Primary training entry point: `cosmos_framework.scripts.train` driven by a structured, pydantic-validated TOML interface (`--sft-toml=<recipe-toml>`); the schema lives at [`cosmos_framework/configs/toml_config/sft_config.py`](./cosmos_framework/configs/toml_config/sft_config.py) and the canonical recipe pattern is documented in [`examples/README.md`](./examples/README.md).

> All paths below are relative to the repository root (the directory containing `pyproject.toml`, the `cosmos_framework/` Python package, and `packages/`).

## Commands

| Task                   | Command                                             |
| ---------------------- | --------------------------------------------------- |
| Lint                   | `uv run ruff check .`                               |
| Format check           | `uv run ruff format --check .`                      |
| Auto-fix lint + format | `uv run ruff check --fix . && uv run ruff format .` |
| Type-check             | `uv run pyrefly check`                              |
| Test (all)             | `uv run pytest`                                     |
| Test (single file)     | `uv run pytest --capture=no <path>`                 |

Config files: `.ruff.toml` (ruff), `pyrefly.toml` (pyrefly), `.pytest.toml` (pytest), `conftest.py` (pytest fixtures).

A `justfile` is provided at the root with longer recipes (`just install`, `just lint`, `just test`, `just docker-cu130`).

## Rules

- Always answer questions with references to code or documentation in `file:line` format.
- When unsure, point the user to the closest doc rather than guessing.
- Keep this file short. Link out to skills and docs for detail — this file is included in every prompt.
- Inference code belongs under `cosmos_framework/inference/`; training infrastructure belongs under the other `cosmos_framework/` subpackages. Don't blur the two — if you find yourself adding training-time imports inside `cosmos_framework/inference/` (or vice versa), reconsider.

## Development and Version-Control Workflow

- Use Git for all source-code, configuration, and documentation changes. Inspect
  the worktree before and after editing, and never include unrelated or generated
  files in a commit.
- GitHub credentials are kept only on the local MacBook. Never copy a token to
  the cluster, write it into a remote URL, store it in the repository or remote
  credential helpers, or expose it in commands, logs, documentation, or commits.
  When authentication is required, pass the credential ephemerally from the
  authorized local workflow.
- After each complete code or feature update, run the relevant validation, create
  a focused Git commit, and push it to the configured remote. Do not leave
  completed implementation work uncommitted unless the user explicitly requests
  that it remain unpublished.
- Every code or feature update must also update the corresponding user,
  developer, setup, or architecture documentation in the same commit. Keep
  examples and commands synchronized with the implementation.
- Store all development-process documentation under [`dev/`](./dev/), relative
  to the repository root.
  This includes development plans, design notes, implementation records,
  experiment notes, review notes, and changelogs. Do not create these documents
  elsewhere in the repository. Root instruction files such as `AGENTS.md` and
  established user-facing or upstream documentation keep their required
  locations and must still be updated in place when affected.
- Store all experiment outputs under
  `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output`. This is the
  single output root for training and inference runs, checkpoints, logs,
  offline W&B data, exported models, visualizations, evaluation artifacts, and
  other generated experiment files. Create per-project and per-run
  subdirectories below it; do not write generated experiment outputs into the
  repository or another shared-storage location.
- Assign every completed change a monotonically increasing identifier in the
  form `DEV-NNNN`. Determine the next identifier from the highest existing entry
  in [`dev/changelog.md`](./dev/changelog.md); never reuse or renumber an ID.
- Start every commit subject with its identifier, for example
  `DEV-0002: add training configuration`, and create exactly one matching
  changelog entry in the same commit. This shared identifier is the authoritative
  mapping between the changelog and Git history.
- Format changelog headings as
  `## DEV-NNNN — YYYY-MM-DD HH:MM +08:00 — Short title`. Keep entries in reverse
  commit order and record separate entries for multiple commits on the same day.
- Include the affected area, concise summary, documentation updated, and
  validation performed. Do not put a commit's own hash in its entry because
  changing the entry would change that hash.
- Documentation-only and development-rule changes follow the same commit and
  changelog sequence.

## Key File Locations

### Training (`cosmos_framework/`)

| What                                                 | Where                                           |
| ---------------------------------------------------- | ----------------------------------------------- |
| Algorithms (losses, RL, reward)                      | `cosmos_framework/algorithm/{loss,reward,rl}`   |
| Training loop                                        | `cosmos_framework/trainer/`                     |
| Models + parallelism                                 | `cosmos_framework/model/`                       |
| Datasets / data loading                              | `cosmos_framework/data/`                        |
| Checkpoint I/O                                       | `cosmos_framework/checkpoint/`                  |
| Callbacks (logging, eval)                            | `cosmos_framework/callbacks/`                   |
| RL workers (rollout, reward, reference, simulations) | `cosmos_framework/workers/`                     |
| Controller / orchestrator                            | `cosmos_framework/controller/`                  |
| Launchers (Slurm, torchrun, k8s)                     | `cosmos_framework/launcher/`                    |
| Evaluation harness                                   | `cosmos_framework/evaluation/`                  |
| CLI tools                                            | `cosmos_framework/tools/`, `tools/` (repo root) |

For a per-subpackage tour with descriptions, see [`docs/code_structure.md`](./docs/code_structure.md).

### Inference (`cosmos_framework/inference/`)

| What                     | Where                                                                            |
| ------------------------ | -------------------------------------------------------------------------------- |
| CLI entry point          | `cosmos_framework/scripts/inference.py`                                          |
| Args / param definitions | `cosmos_framework/inference/args.py`                                             |
| Per-modality defaults    | `cosmos_framework/inference/defaults/<mode>/sample_args.json`                    |
| Model / inference core   | `cosmos_framework/inference/model.py`, `cosmos_framework/inference/inference.py` |
| Ray serving              | `cosmos_framework/inference/ray/`                                                |
| Backend packages         | `packages/{diffusers,transformers,vllm}-cosmos3/`                                |
| Example inputs           | `inputs/omni/*.json`, `inputs/reasoner/*.json`                                   |

## Documentation

| Doc                                                | What it covers                                                    |
| -------------------------------------------------- | ----------------------------------------------------------------- |
| [docs/setup.md](./docs/setup.md)                   | Install, NGC base image, CUDA variants, base-checkpoint download. |
| [docs/code_structure.md](./docs/code_structure.md) | Repo layout and per-subpackage tour of `cosmos_framework/`.       |
| [docs/training.md](./docs/training.md)             | Single- and multi-node launches, parallelism, mixed precision.    |
| [docs/inference.md](./docs/inference.md)           | Sample arguments, parallelism, schemas, troubleshooting.          |
| [docs/faq.md](./docs/faq.md)                       | Troubleshooting (OOM, NCCL, slow training) + env vars.            |

Agent skills (codebase navigation, env troubleshooting, inference, post-training, setup) live in [`.agents/skills/`](./.agents/skills) and [`.claude/skills/`](./.claude/skills).

## Common Tasks

### Training

| Task                     | Command                                                                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Single-GPU train (smoke) | `python -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml`                                                           |
| Multi-GPU train          | `IMAGINAIRE_OUTPUT_ROOT=outputs/train torchrun --nproc-per-node=8 -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml` |
| Resume from checkpoint   | Re-run the same `train --sft-toml=<recipe>.toml` against the same `IMAGINAIRE_OUTPUT_ROOT` (auto-resume from latest DCP).                              |
| Export DCP → HF          | `python -m cosmos_framework.scripts.export_model --src <dcp> --dst <hf>`                                                                               |
| Run a config sweep       | `just run python -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/<recipe>.toml -- key.path=value ...`                            |

### Inference

| Task                    | Command                                                                                                                           |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Single-GPU inference    | `python -m cosmos_framework.scripts.inference -i inputs/omni/t2v.json -o outputs/ --checkpoint-path Cosmos3-Nano`                 |
| Multi-GPU inference     | `torchrun --nproc-per-node=4 -m cosmos_framework.scripts.inference --parallelism-preset=latency -i ... -o outputs/ ...`           |
| Start online Ray server | `python -m cosmos_framework.inference.ray.serve --parallelism-preset=latency -o outputs/ray_serve --checkpoint-path Cosmos3-Nano` |
| Launch Gradio UI        | `python -m cosmos_framework.inference.ray.gradio --port=8080`                                                                     |
| See all CLI flags       | `python -m cosmos_framework.scripts.inference --help`                                                                             |

## Gotchas

- **NGC / PyTorch containers**: run `export LD_LIBRARY_PATH=''` before any `python` call or you'll hit a `torch._C` import error. See [`docs/setup.md`](./docs/setup.md#pytorch-import-issue).
- **Reproducibility**: always pass `--seed <int>`. Without it a random seed is used each run.
- **JSON paths**: relative paths inside input JSON files resolve relative to the JSON file's directory, not the working directory.
- **Resume**: re-running the same inference command skips already-generated outputs automatically.
- **Separation of concerns**: keep training-time imports out of `cosmos_framework/inference/`, and keep heavyweight inference-only deps (vLLM, Ray Serve, Gradio) gated behind optional extras so plain training installs stay slim.
