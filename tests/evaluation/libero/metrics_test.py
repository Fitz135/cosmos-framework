# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest

from cosmos_framework.evaluation.libero.metrics import aggregate_episode_metrics, wilson_interval


def test_wilson_interval_known_value_and_empty_sample() -> None:
    lower, upper = wilson_interval(5, 10)
    assert lower == pytest.approx(0.236593, abs=1e-6)
    assert upper == pytest.approx(0.763407, abs=1e-6)
    assert wilson_interval(0, 0) == (None, None)


@pytest.mark.parametrize("successes,total", [(-1, 1), (2, 1), (0, -1)])
def test_wilson_interval_rejects_invalid_counts(successes: int, total: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, total)


def test_aggregate_tasks_suites_overall_and_separates_infra_errors() -> None:
    episodes = [
        {"task_suite": "libero_goal", "task_id": 0, "success": True},
        {"task_suite": "libero_goal", "task_id": 0, "success": False},
        {"task_suite": "libero_goal", "task_id": 1, "success": True},
        {
            "task_suite": "libero_goal",
            "task_id": 1,
            "infra_error": True,
            "infra_error_type": "server_timeout",
        },
        {"task_suite": "libero_object", "task_id": 0, "success": False},
        {
            "task_suite": "libero_object",
            "task_id": 0,
            "success": False,
            "infra_error": True,
            "error_type": "egl_context",
        },
    ]

    metrics = aggregate_episode_metrics(episodes)
    tasks = {(row["task_suite"], row["task_id"]): row for row in metrics["tasks"]}
    suites = {row["task_suite"]: row for row in metrics["suites"]}

    assert tasks[("libero_goal", 0)]["success_rate"] == 0.5
    assert tasks[("libero_goal", 1)]["attempted_episodes"] == 2
    assert tasks[("libero_goal", 1)]["evaluated_episodes"] == 1
    assert tasks[("libero_goal", 1)]["success_rate"] == 1.0
    assert suites["libero_goal"]["success_rate"] == pytest.approx(2 / 3)
    assert suites["libero_goal"]["macro_task_success_rate"] == 0.75
    assert suites["libero_object"]["success_rate"] == 0.0

    overall = metrics["overall"]
    assert overall["attempted_episodes"] == 6
    assert overall["evaluated_episodes"] == 4
    assert overall["successes"] == 2
    assert overall["failures"] == 2
    assert overall["infra_errors"] == 2
    assert overall["success_rate"] == 0.5
    assert overall["macro_task_success_rate"] == pytest.approx(0.5)
    assert overall["macro_suite_success_rate"] == pytest.approx(1 / 3)
    assert metrics["infrastructure_errors"] == {
        "count": 2,
        "by_suite": {"libero_goal": 1, "libero_object": 1},
        "by_type": {"egl_context": 1, "server_timeout": 1},
    }


def test_task_with_only_infra_errors_has_no_rate() -> None:
    metrics = aggregate_episode_metrics(
        [{"task_suite": "libero_10", "task_id": 9, "infra_error": True, "infra_error_type": "server"}]
    )

    task = metrics["tasks"][0]
    assert task["attempted_episodes"] == 1
    assert task["evaluated_episodes"] == 0
    assert task["success_rate"] is None
    assert task["wilson_95_ci"] == {"lower": None, "upper": None}
    assert metrics["overall"]["success_rate"] is None


def test_aggregate_rejects_missing_success_for_non_infra_record() -> None:
    with pytest.raises(ValueError, match="boolean 'success'"):
        aggregate_episode_metrics([{"task_suite": "libero_10", "task_id": 0}])


def test_empty_aggregation_is_well_defined() -> None:
    metrics = aggregate_episode_metrics([])
    assert metrics["tasks"] == []
    assert metrics["suites"] == []
    assert metrics["overall"]["evaluated_episodes"] == 0
    assert metrics["overall"]["success_rate"] is None
    assert metrics["infrastructure_errors"]["count"] == 0
