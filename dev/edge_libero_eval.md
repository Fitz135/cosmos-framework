# Cosmos3-Edge LIBERO Eval Runbook

本页用于在 PJLab H 集群复现 Cosmos3-Edge 的严格 LIBERO 闭环评测。正式结果必须使用独立的
checkpoint × suite 目录，不能把不同 checkpoint 的 episode 混合统计。

## 1. 代码入口

| 入口 | 作用 |
| ---- | ---- |
| `cosmos_framework/evaluation/libero/preflight.py` | 检查 LIBERO/robosuite/MuJoCo、资产、EGL、四个 suite 和 adapter/stats。 |
| `cosmos_framework/evaluation/libero/job.py` | 解析 checkpoint 身份，启动 server/runner，校验握手并清理进程组。 |
| `cosmos_framework/evaluation/libero/runner.py` | 批量闭环 rollout、确定性 seed、断点续跑和 sealed artifacts。 |
| `cosmos_framework/scripts/action_policy_server_libero.py` | CUDA policy server，提供 `/info` 与批量动作预测。 |
| `cosmos_framework/evaluation/libero/aggregate.py` | 重新校验 sealed runs，聚合 task/suite/checkpoint 并原子写 summary。 |

严格协议为 `cosmos-libero-eval-v3`；目标 adapter 是
`cosmos_framework/evaluation/libero/profiles/edge_libero_target_adapter.json`。

## 2. 环境与 preflight

以下 GPU 命令在 `rlaunch`/RJob worker 内执行，不要在 workspace 开发机直接运行。

```bash
source /mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/.bashrc
proxy_on
REPO=/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/project/cosmos-framework
OUT=/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-eval
SERVER_PYTHON="$REPO/.venv/bin/python"
RUNNER_PYTHON="$REPO/.venv-libero/bin/python"
ADAPTER="$REPO/cosmos_framework/evaluation/libero/profiles/edge_libero_target_adapter.json"
export LIBERO_CONFIG_PATH="$OUT/runtime/libero-config"
export LD_LIBRARY_PATH='' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0 EGL_DEVICE_ID=0
cd "$REPO"

"$RUNNER_PYTHON" -m cosmos_framework.evaluation.libero.preflight \
  --output-dir "$OUT/libero/preflight/new-preflight-run" --seed 0 --render-gpu-device-id 0
```

仅当 `preflight.json` 的 `status` 为 `passed` 才进入评测。已验证环境为 LIBERO 0.1.1、
robosuite 1.4.0、MuJoCo 3.3.2 和 EGL。

## 3. Suites 与 rollout 规模

| Suite | Task IDs | Canonical `max_steps` | 正式规模 |
| ----- | -------- | --------------------- | -------- |
| `libero_spatial` | 0–9 | 220 | 10 × 50 = 500 episodes |
| `libero_object` | 0–9 | 280 | 10 × 50 = 500 episodes |
| `libero_goal` | 0–9 | 300 | 10 × 50 = 500 episodes |
| `libero_10` | 0–9 | 520 | 10 × 50 = 500 episodes |

传 `--task-ids ''` 选择全部 10 个 tasks；传 `--max-steps 0` 使用上表 canonical 值。

## 4. 本地/交互 worker 运行一个 suite

先按 checkpoint 设置 `CHECKPOINT_ID`、`CHECKPOINT`、`WEIGHTS_VARIANT`；5k 还需
`CONFIG_ARGS=(--config-file "$OUT/checkpoints/cosmos3-edge-libero-5k-ema-runtime/config.json")`，
Base/10k 使用 `CONFIG_ARGS=()`。

```bash
RUN_ID=new-full-run-id
SUITE=libero_spatial
RUN_DIR="$OUT/libero/runs/$CHECKPOINT_ID/$RUN_ID/$SUITE"
export COSMOS_EVAL_IMAGE=registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_f2adb47
export COSMOS_EVAL_JOB_ID=local-edge-libero-debug

"$SERVER_PYTHON" -m cosmos_framework.evaluation.libero.job \
  --checkpoint-path "$CHECKPOINT" "${CONFIG_ARGS[@]}" \
  --target-adapter-path "$ADAPTER" --weights-variant "$WEIGHTS_VARIANT" \
  --runner-python "$RUNNER_PYTHON" --server-port 8000 \
  --num-steps 8 --guidance 1.0 --task-suite "$SUITE" --task-ids '' --trials 50 \
  --num-envs 8 --action-horizon 8 --seed 0 --max-steps 0 --warmup-steps 10 \
  --mujoco-gl egl --render-gpu-device-id 0 --request-timeout 120 --output-dir "$RUN_DIR"
```

## 5. 已验证的 RJob 骨架

一个 RJob 只运行一个 checkpoint × suite；对每个 checkpoint 执行一次四-suite 循环。
先设置该 checkpoint 的 `CHECKPOINT_ID/JOB_TAG/CHECKPOINT/WEIGHTS_VARIANT`；Base/10k 的
`CONFIG_FLAG=''`，5k 使用 `CONFIG_FLAG="--config-file $OUT/checkpoints/cosmos3-edge-libero-5k-ema-runtime/config.json"`。

```bash
source /etc/profile.d/ssh-init.sh
IMAGE=registry.h.pjlab.org.cn/ailab-llmrazor/xtuner_tmp:pt28_20260303_f2adb47
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  JOB_NAME="c3-libero-${JOB_TAG}-${SUITE#libero_}-${RUN_ID}"
  RUN_DIR="$OUT/libero/runs/$CHECKPOINT_ID/$RUN_ID/$SUITE"
  JOB_COMMAND="source /mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/.bashrc >/dev/null;
    cd $REPO;
    export LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH LD_LIBRARY_PATH='' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl;
    export MUJOCO_EGL_DEVICE_ID=0 EGL_DEVICE_ID=0 COSMOS_EVAL_IMAGE=$IMAGE COSMOS_EVAL_JOB_ID=$JOB_NAME;
    exec $SERVER_PYTHON -m cosmos_framework.evaluation.libero.job \
      --checkpoint-path $CHECKPOINT $CONFIG_FLAG --target-adapter-path $ADAPTER \
      --weights-variant $WEIGHTS_VARIANT --runner-python $RUNNER_PYTHON --server-port 8000 \
      --num-steps 8 --guidance 1.0 --task-suite $SUITE --task-ids '' --trials 50 \
      --num-envs 8 --action-horizon 8 --seed 0 --max-steps 0 --warmup-steps 10 \
      --mujoco-gl egl --render-gpu-device-id 0 --request-timeout 120 --output-dir $RUN_DIR"
  rjob submit --name "$JOB_NAME" \
    --group evoagi_gpu --charged-group evoagi_gpu \
    --private-machine group --preemptible no --image "$IMAGE" \
    --replica 1 --gpu 1 --cpu 16 --memory 131072 \
    --positive-tags feature/gpfs=yes --custom-resources brainpp.cn/fuse=1 \
    --mount gpfs://gpfs1/evoagi-share/VTLA:/mnt/shared-storage-user/evoagi-share/VTLA \
    --share-host-shm true --restart-policy never --backoff_limit 1 \
    --auto-delete-duration 168h -- bash -lc "$JOB_COMMAND"
done
```

不要添加推测的 `h200` positive tag；它曾排除可用 H200。监控使用
`rjob get <JOB>`、`rjob events <REPLICA> --replica`、
`rjob logs replica <REPLICA> -n 200` 和 `rjob logs job <JOB> -n 200`。

## 6. 产物与结果检查

每个 sealed run 的核心产物是 `manifest.json`、`episodes.jsonl`、`metrics.json`、
`_SUCCESS`、`job.log`、`server.log`、`runner.log` 和 `server_runtime/`。
`infra_errors.jsonl` 只在出现过历史 infra attempt 时存在；文件缺失表示 0，属于合法状态。
正式 suite 应满足：

```bash
test -f "$RUN_DIR/_SUCCESS"
test "$(wc -l < "$RUN_DIR/episodes.jsonl")" -eq 500
test ! -s "$RUN_DIR/infra_errors.jsonl"
"$RUNNER_PYTHON" -m json.tool "$RUN_DIR/metrics.json" | less
tail -f "$RUN_DIR/job.log" "$RUN_DIR/server.log" "$RUN_DIR/runner.log"
```

正式三-checkpoint summary 用 `aggregate` 读取同一 run ID 的 12 个 sealed 目录；命令见
`docs/action_policy_libero_posttrain.md`。查看方法：

```bash
REPORT="$OUT/libero/reports/full-v1-45ebff5/summary.json"
sha256sum "$REPORT"
"$SERVER_PYTHON" -m json.tool "$REPORT" | less
```

## 7. 本次正式评测

Run ID：`full-v1-45ebff5`；report：
`/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/output/cosmos-framework-eval/libero/reports/full-v1-45ebff5/summary.json`。
报告 SHA-256：`0e5a5fa4090cae396a227516175bd1e5ef36675035f4593a8b5298fbddad76d2`。

| Checkpoint | 路径/身份 | Spatial | Object | Goal | LIBERO-10 | Overall |
| ---------- | --------- | ------- | ------ | ---- | --------- | ------- |
| Base Edge regular | `$OUT/checkpoints/cosmos3-edge-base-regular-hf`，zero-shot | 0.0% | 0.0% | 0.0% | 0.0% | 0/2000 (0.00%) |
| 5k Edge EMA | `/mnt/shared-storage-user/evoagi-share/VTLA/lutianyi/model/cosmos3-edge-libero-all-10fps/export-final-iter5000` | 77.6% | 95.2% | 79.4% | 82.0% | 1671/2000 (83.55%) |
| 10k Edge EMA | `$OUT/checkpoints/cosmos3-edge-libero-10k-ema` | 76.8% | 95.8% | 74.2% | 85.0% | 1659/2000 (82.95%) |

12/12 RJobs 均成功；共 6000 个 terminal episodes，infra attempts/errors 均为 0。
