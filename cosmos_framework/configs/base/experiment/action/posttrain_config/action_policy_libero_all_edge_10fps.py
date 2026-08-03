# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge action-policy SFT on a merged, 10 FPS LIBERO dataset.

The recipe preserves the pretrained Edge action heads and trains all generation
and action modules on 8-action (0.8 second) chunks. ``LIBERO_ROOT`` is one
merged LeRobot v3 root with the external camera in
``observation.images.image`` and the wrist camera in
``observation.images.image2``. See ``docs/action_policy_libero_posttrain.md``.
"""

import copy
import os

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.every_n_draw_sample import EveryNDrawSample
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_libero_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _action_policy_libero_edge_model_config() -> dict:
    """Return an isolated Edge config for LIBERO action post-training."""
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    # The base DCP supplies every model tensor, including the renewed Edge
    # action heads. Do not separately initialize the diffusion expert.
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    cfg["tokenizer"]["encode_exact_durations"] = None
    if processor_path := os.environ.get("COSMOS3_EDGE_PROCESSOR_PATH"):
        tokenizer_cfg = cfg["vlm_config"]["tokenizer"]
        tokenizer_cfg["repository"] = None
        tokenizer_cfg["revision"] = None
        tokenizer_cfg["subdir"] = ""
        tokenizer_cfg["tokenizer_type"] = processor_path
    return cfg


action_policy_libero_all_edge_10fps = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
            # diverged on the action loss).
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},  # linear LR decay
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_libero",
            group="action_sft",
            name="action_policy_libero_all_edge_10fps",
            wandb_mode="offline",
        ),
        model=dict(
            config=_action_policy_libero_edge_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,  # popped by build_optimizer for FusedAdam (fused by construction)
            # Train the generation + action heads.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "k_norm_und_for_gen",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=5.0e-05,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke: 100 iters (real run sets via TOML, GA=10000)
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[0],  # smoke (real run sets via TOML, GA=2000)
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,  # real run sets via TOML (GA=2)
            logging_iter=1,
            max_iter=100,  # smoke
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                # The trainer saves periodic checkpoints before on_training_step_end,
                # so this produces one qualitative EMA rollout after every periodic save.
                libero_rollout=L(EveryNDrawSample)(
                    every_n=2000,
                    n_viz_sample=1,
                    n_sample_to_save=1,
                    num_sampling_step=8,
                    guidance=[1.0],
                    do_x0_prediction=False,
                    save_s3=False,
                    save_local=True,
                    is_ema=True,
                    fps=10,
                ),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # EMA warm-starts from net. Everything else, including all renewed
            # Edge action-head tensors, must exist in the base DCP.
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Edge DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            # Periodic saves are every 2000 steps. The trainer also saves max_iter
            # when it is not divisible by this interval.
            save_iter=2000,
            strict_resume=True,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_libero_all_edge_10fps",
            max_samples_per_batch=128,  # global = 128 x FSDP8 x grad_accum2 = 2048
            max_sequence_length=None,  # None disables token packing (TOML can't express null)
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                # The merged dataset contains all 40 tasks in one LeRobot root.
                # Episode-block shuffling is deterministic and each worker/rank
                # receives disjoint blocks.
                datasets=dict(
                    libero_all_10fps=dict(
                        ratio=1,
                        dataset=L(get_action_libero_sft_dataset)(
                            root="${oc.env:LIBERO_ROOT}",
                            embodiment_type="libero",
                            fps=10,
                            chunk_length=8,
                            image_size=256,  # concat_view -> 256x512
                            mode="wam",
                            camera_mode="concat_view",
                            wrist_camera_key="observation.images.image2",
                            video_backend="pyav",
                            action_space="frame_wise_relative",
                            rotation_space="6d",
                            pose_coordinate_frame="native",
                            action_normalization="quantile_rot",
                            action_stats_path=("libero_10fps_pm_one_native_frame_wise_relative_rot6d.json"),
                            split="full",
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            resolution=None,
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            format_prompt_as_json=True,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [action_policy_libero_all_edge_10fps]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
