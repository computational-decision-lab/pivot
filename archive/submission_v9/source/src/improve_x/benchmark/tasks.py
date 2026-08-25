from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from .dataset import ImprovementBenchRow


def _sign(value: float | None, tolerance: float = 1e-9) -> int:
    if value is None or abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def evaluate_sign_task(
    rows: Sequence[ImprovementBenchRow], predictions: Mapping[str, float]
) -> dict[str, float | int | None]:
    comparable = [row for row in rows if row.deployment_delta is not None and row.transition_id in predictions]
    comparable = [row for row in comparable if _sign(row.deployment_delta) != 0]
    correct = sum(_sign(row.deployment_delta) == _sign(float(predictions[row.transition_id])) for row in comparable)
    return {
        "accuracy": correct / len(comparable) if comparable else None,
        "n": len(comparable),
        "correct": correct,
    }


def evaluate_ranking_task(
    rows: Sequence[ImprovementBenchRow], scores: Mapping[str, float]
) -> dict[str, float | int | None]:
    groups: dict[tuple[object, ...], list[ImprovementBenchRow]] = defaultdict(list)
    for row in rows:
        if row.deployment_delta is not None and row.transition_id in scores:
            # A round can contain the same candidate transition evaluated in
            # several worlds. Keep those candidate pools separate, but pool
            # update operators that competed from the same incumbent.
            incumbent_key = tuple(sorted(row.incumbent_policy.items()))
            trajectory_id = row.metadata.get("trajectory_id")
            if trajectory_id is None:
                trajectory_id = f"seed:{row.seed}"
            groups[(row.world_level, str(trajectory_id), row.round_id, incumbent_key)].append(row)
    correct = 0
    for candidates in groups.values():
        if not candidates:
            continue
        selected = max(candidates, key=lambda row: float(scores[row.transition_id]))
        oracle = max(candidates, key=lambda row: row.deployment_delta or float("-inf"))
        correct += int(selected.transition_id == oracle.transition_id)
    return {"accuracy": correct / len(groups) if groups else None, "n_groups": len(groups), "correct": correct}


def evaluate_explanation_task(
    rows: Sequence[ImprovementBenchRow], predictions: Mapping[str, str]
) -> dict[str, float | int | None]:
    labels = sorted({row.failure_type for row in rows if row.transition_id in predictions})
    per_class: list[float] = []
    correct = 0
    total = 0
    for label in labels:
        expected = [row for row in rows if row.failure_type == label and row.transition_id in predictions]
        if not expected:
            continue
        hits = sum(predictions[row.transition_id] == label for row in expected)
        correct += hits
        total += len(expected)
        per_class.append(hits / len(expected))
    return {
        "macro_accuracy": sum(per_class) / len(per_class) if per_class else None,
        "accuracy": correct / total if total else None,
        "n": total,
        "n_classes": len(per_class),
    }
