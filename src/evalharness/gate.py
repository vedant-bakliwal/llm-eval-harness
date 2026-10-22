"""The CI quality gate: absolute thresholds plus a no-regression check against a baseline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    passed: bool
    failures: list[str]


def check(metrics: dict[str, float], thresholds: dict[str, float], baseline: dict[str, float] | None = None) -> GateResult:
    failures = []
    for name, minimum in thresholds.items():
        if name == "max_regression" or name not in metrics:
            continue  # judge metrics are only checked when a judge ran
        if metrics[name] < minimum:
            failures.append(f"{name} = {metrics[name]:.2f}, below the threshold {minimum:.2f}")
    if baseline:
        allowed = thresholds.get("max_regression", 0.0)
        for name, before in baseline.items():
            now = metrics.get(name)
            if now is not None and before - now > allowed + 1e-9:
                failures.append(f"{name} regressed from {before:.2f} to {now:.2f} (allowed drop {allowed:.2f})")
    return GateResult(not failures, failures)
