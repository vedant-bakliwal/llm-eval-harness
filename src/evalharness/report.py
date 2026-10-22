"""Markdown report for humans; JSON for machines and baselines."""
from __future__ import annotations

from .gate import GateResult
from .metrics import CaseResult


def render(label: str, metrics: dict[str, float], gate: GateResult, results: list[CaseResult],
           thresholds: dict[str, float], judge: str) -> str:
    lines = [f"# Eval report: {label}", "", f"## Quality gate: {'PASSED' if gate.passed else 'FAILED'}", ""]
    lines += [f"- {f}" for f in gate.failures]
    if gate.failures:
        lines.append("")
    lines += ["| Metric | Score | Threshold |", "|---|---|---|"]
    for name, value in metrics.items():
        t = thresholds.get(name)
        lines.append(f"| {name} | {value:.2f} | {'' if t is None else f'{t:.2f}'} |")
    if judge == "none":
        lines += ["", "LLM-as-judge metrics (faithfulness, answer relevance) were not run. Use `--judge bedrock` or `--judge deepeval`."]
    failing = [r for r in results if r.failed_metrics]
    lines += ["", f"## Failing cases ({len(failing)} of {len(results)})", ""]
    if not failing:
        lines.append("None.")
    else:
        lines += ["| Case | Category | Failed metrics | Answer |", "|---|---|---|---|"]
        for r in failing:
            answer = r.answer.text.replace("|", "/").replace("\n", " ")[:120]
            lines.append(f"| {r.case.id} | {r.case.category} | {', '.join(r.failed_metrics)} | {answer} |")
    return "\n".join(lines) + "\n"
