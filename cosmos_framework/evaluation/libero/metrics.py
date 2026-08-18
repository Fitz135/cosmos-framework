# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Success-rate aggregation for LIBERO closed-loop evaluation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from cosmos_framework.evaluation.libero.action import GRIPPER_ADAPTER_CONTRACT

_WILSON_95_Z = 1.959963984540054


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    """Return the two-sided 95% Wilson score interval for a binomial rate."""

    if isinstance(successes, bool) or isinstance(total, bool):
        raise TypeError("successes and total must be integers, not booleans")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise TypeError("successes and total must be integers")
    if total < 0 or successes < 0 or successes > total:
        raise ValueError(f"Expected 0 <= successes <= total, got successes={successes}, total={total}")
    if total == 0:
        return None, None

    probability = successes / total
    z_squared = _WILSON_95_Z**2
    denominator = 1.0 + z_squared / total
    center = (probability + z_squared / (2.0 * total)) / denominator
    margin = (
        _WILSON_95_Z * math.sqrt(probability * (1.0 - probability) / total + z_squared / (4.0 * total**2)) / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


@dataclass
class _Counts:
    attempted: int = 0
    successes: int = 0
    failures: int = 0
    infra_errors: int = 0

    @property
    def evaluated(self) -> int:
        return self.successes + self.failures

    def add(self, *, success: bool | None, infra_error: bool) -> None:
        self.attempted += 1
        if infra_error:
            self.infra_errors += 1
        elif success:
            self.successes += 1
        else:
            self.failures += 1


def _rate_fields(counts: _Counts) -> dict[str, Any]:
    lower, upper = wilson_interval(counts.successes, counts.evaluated)
    return {
        "attempted_episodes": counts.attempted,
        "evaluated_episodes": counts.evaluated,
        "successes": counts.successes,
        "failures": counts.failures,
        "infra_errors": counts.infra_errors,
        "success_rate": counts.successes / counts.evaluated if counts.evaluated else None,
        "wilson_95_ci": {"lower": lower, "upper": upper},
    }


def _require_suite(record: Mapping[str, Any], index: int) -> str:
    suite = record.get("task_suite")
    if not isinstance(suite, str) or not suite.strip():
        raise ValueError(f"Episode record {index} must contain a non-empty string 'task_suite'")
    return suite


def _require_task_id(record: Mapping[str, Any], index: int) -> str | int:
    task_id = record.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, (str, int)):
        raise ValueError(f"Episode record {index} must contain a string or integer 'task_id'")
    if isinstance(task_id, str) and not task_id.strip():
        raise ValueError(f"Episode record {index} contains an empty 'task_id'")
    return task_id


def _infra_error(record: Mapping[str, Any], index: int) -> bool:
    value = record.get("infra_error", False)
    if not isinstance(value, bool):
        raise ValueError(f"Episode record {index} field 'infra_error' must be boolean")
    return value


def _success(record: Mapping[str, Any], *, infra_error: bool, index: int) -> bool | None:
    if infra_error:
        return None
    value = record.get("success")
    if not isinstance(value, bool):
        raise ValueError(f"Non-infrastructure episode record {index} must contain boolean 'success'")
    return value


def _infra_error_type(record: Mapping[str, Any]) -> str:
    value = record.get("infra_error_type", record.get("error_type", "unspecified"))
    if value is None:
        return "unspecified"
    return str(value)


def aggregate_episode_metrics(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate episode outcomes at task, suite, and overall levels.

    Infrastructure errors are counted in ``attempted_episodes`` and reported
    separately, but excluded from ``evaluated_episodes`` and every success-rate
    denominator.  Policy/environment failures therefore remain explicit
    ``success=False`` records with ``infra_error=False``.
    """

    task_counts: dict[tuple[str, str | int], _Counts] = {}
    suite_counts: dict[str, _Counts] = {}
    overall_counts = _Counts()
    infra_by_suite: Counter[str] = Counter()
    infra_by_type: Counter[str] = Counter()

    for index, record in enumerate(episodes):
        if not isinstance(record, Mapping):
            raise TypeError(f"Episode record {index} must be a mapping")
        suite = _require_suite(record, index)
        task_id = _require_task_id(record, index)
        is_infra_error = _infra_error(record, index)
        success = _success(record, infra_error=is_infra_error, index=index)

        task_counts.setdefault((suite, task_id), _Counts()).add(success=success, infra_error=is_infra_error)
        suite_counts.setdefault(suite, _Counts()).add(success=success, infra_error=is_infra_error)
        overall_counts.add(success=success, infra_error=is_infra_error)
        if is_infra_error:
            infra_by_suite[suite] += 1
            infra_by_type[_infra_error_type(record)] += 1

    tasks: list[dict[str, Any]] = []
    for (suite, task_id), counts in sorted(task_counts.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        tasks.append({"task_suite": suite, "task_id": task_id, **_rate_fields(counts)})

    suites: list[dict[str, Any]] = []
    for suite, counts in sorted(suite_counts.items()):
        task_rates = [
            task["success_rate"] for task in tasks if task["task_suite"] == suite and task["success_rate"] is not None
        ]
        suites.append(
            {
                "task_suite": suite,
                "num_tasks": sum(task["task_suite"] == suite for task in tasks),
                "macro_task_success_rate": fmean(task_rates) if task_rates else None,
                **_rate_fields(counts),
            }
        )

    task_rates = [task["success_rate"] for task in tasks if task["success_rate"] is not None]
    suite_rates = [suite["success_rate"] for suite in suites if suite["success_rate"] is not None]
    overall = {
        "num_suites": len(suites),
        "num_tasks": len(tasks),
        "macro_task_success_rate": fmean(task_rates) if task_rates else None,
        "macro_suite_success_rate": fmean(suite_rates) if suite_rates else None,
        **_rate_fields(overall_counts),
    }

    return {
        "confidence_level": 0.95,
        "tasks": tasks,
        "suites": suites,
        "overall": overall,
        "infrastructure_errors": {
            "count": overall_counts.infra_errors,
            "by_suite": dict(sorted(infra_by_suite.items())),
            "by_type": dict(sorted(infra_by_type.items())),
        },
    }


def _gripper_count(value: Any, name: str, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Episode record {index} gripper telemetry field {name!r} must be a non-negative integer")
    return value


def _gripper_float(value: Any, name: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Episode record {index} gripper telemetry field {name!r} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Episode record {index} gripper telemetry field {name!r} must be a finite number")
    return result


def aggregate_gripper_adapter_metrics(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate generated-chunk clamp telemetry from terminal episodes.

    Counts are summed before computing the rate; episode-level rates are never
    averaged. Infrastructure attempts are excluded even when their records
    contain partial adapter telemetry.
    """
    generated_count = 0
    clipped_count = 0
    raw_min: float | None = None
    raw_max: float | None = None

    for index, record in enumerate(episodes):
        if not isinstance(record, Mapping):
            raise TypeError(f"Episode record {index} must be a mapping")
        if _infra_error(record, index):
            continue
        _success(record, infra_error=False, index=index)
        if record.get("gripper_adapter_contract") != GRIPPER_ADAPTER_CONTRACT:
            raise ValueError(f"Episode record {index} must use gripper_adapter_contract={GRIPPER_ADAPTER_CONTRACT!r}")
        telemetry = record.get("gripper_adapter_telemetry")
        if not isinstance(telemetry, Mapping):
            raise ValueError(f"Episode record {index} must contain mapping 'gripper_adapter_telemetry'")

        episode_generated = _gripper_count(telemetry.get("generated_value_count"), "generated_value_count", index)
        episode_clipped = _gripper_count(
            telemetry.get("clipped_generated_value_count"), "clipped_generated_value_count", index
        )
        if episode_clipped > episode_generated:
            raise ValueError(
                f"Episode record {index} clipped_generated_value_count must not exceed generated_value_count"
            )

        episode_raw_min = telemetry.get("raw_min")
        episode_raw_max = telemetry.get("raw_max")
        episode_rate = telemetry.get("clipped_generated_value_rate")
        episode_overshoot = telemetry.get("max_abs_overshoot")
        if episode_generated == 0:
            if episode_clipped != 0 or any(
                value is not None for value in (episode_raw_min, episode_raw_max, episode_rate, episode_overshoot)
            ):
                raise ValueError(f"Episode record {index} zero-count gripper telemetry must use null derived values")
        else:
            checked_min = _gripper_float(episode_raw_min, "raw_min", index)
            checked_max = _gripper_float(episode_raw_max, "raw_max", index)
            if checked_min > checked_max:
                raise ValueError(f"Episode record {index} gripper telemetry raw_min must not exceed raw_max")
            expected_clipped = checked_min < -1.0 or checked_max > 1.0
            if expected_clipped != (episode_clipped > 0):
                raise ValueError(f"Episode record {index} gripper telemetry range and clipped count are inconsistent")
            checked_rate = _gripper_float(episode_rate, "clipped_generated_value_rate", index)
            expected_rate = episode_clipped / episode_generated
            if not math.isclose(checked_rate, expected_rate, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"Episode record {index} gripper telemetry clipped rate is inconsistent with counts")
            checked_overshoot = _gripper_float(episode_overshoot, "max_abs_overshoot", index)
            expected_overshoot = max(0.0, -1.0 - checked_min, checked_max - 1.0)
            if not math.isclose(checked_overshoot, expected_overshoot, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"Episode record {index} gripper telemetry max_abs_overshoot is inconsistent")
            raw_min = checked_min if raw_min is None else min(raw_min, checked_min)
            raw_max = checked_max if raw_max is None else max(raw_max, checked_max)

        generated_count += episode_generated
        clipped_count += episode_clipped

    max_abs_overshoot = max(0.0, -1.0 - raw_min, raw_max - 1.0) if raw_min is not None and raw_max is not None else None
    return {
        "contract": GRIPPER_ADAPTER_CONTRACT,
        "generated_value_count": generated_count,
        "clipped_generated_value_count": clipped_count,
        "clipped_generated_value_rate": clipped_count / generated_count if generated_count else None,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "max_abs_overshoot": max_abs_overshoot,
    }
