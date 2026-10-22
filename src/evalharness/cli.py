"""Command line: run the golden set, score it, apply the gate, write reports.

    python -m evalharness run                              # offline generator, no judge
    python -m evalharness run --generator bedrock --judge bedrock
    python -m evalharness run --variant unsafe             # shows the gate blocking a regressed build
    python -m evalharness run --write-baseline             # record current scores as the baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import gate, report
from .corpus import load_corpus
from .generators import Assistant, BedrockGenerator, ExtractiveGenerator
from .metrics import CaseResult, aggregate, load_cases, score_case
from .retriever import BM25Retriever


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalharness", description="Evaluate a RAG assistant and gate releases on quality.")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the evaluation")
    run.add_argument("--corpus", default="data/handbook")
    run.add_argument("--cases", default="evals/golden.jsonl")
    run.add_argument("--thresholds", default="evals/thresholds.json")
    run.add_argument("--baseline", default="evals/baseline.json")
    run.add_argument("--generator", choices=["offline", "bedrock"], default="offline")
    run.add_argument("--variant", choices=["safe", "unsafe"], default="safe",
                     help="offline generator only: 'unsafe' simulates a release with guardrails removed")
    run.add_argument("--judge", choices=["none", "bedrock", "deepeval"], default="none")
    run.add_argument("--out", default="out")
    run.add_argument("--write-baseline", action="store_true")
    return p


def evaluate(args: argparse.Namespace) -> tuple[dict[str, float], gate.GateResult, list[CaseResult], str]:
    chunks = load_corpus(Path(args.corpus))
    if args.generator == "bedrock":
        generator, label = BedrockGenerator(), "bedrock"
    else:
        generator, label = ExtractiveGenerator(safe=args.variant == "safe"), f"offline-{args.variant}"
    assistant = Assistant(BM25Retriever(chunks), generator)

    judge = None
    if args.judge == "bedrock":
        from .judges.bedrock_judge import BedrockJudge
        judge = BedrockJudge()
    elif args.judge == "deepeval":
        from .judges.deepeval_judge import DeepEvalJudge
        judge = DeepEvalJudge()

    results = []
    for case in load_cases(Path(args.cases)):
        answer = assistant.ask(case.question)
        result = CaseResult(case, answer, score_case(case, answer))
        if judge is not None:
            result.judge, result.judge_reason = judge.score(case.question, answer.text, answer.contexts)
        results.append(result)

    metrics = aggregate(results)
    thresholds = json.loads(Path(args.thresholds).read_text())
    baseline_path = Path(args.baseline)
    baseline = None
    if baseline_path.exists() and not args.write_baseline:
        baseline = json.loads(baseline_path.read_text()).get("metrics")
    return metrics, gate.check(metrics, thresholds, baseline), results, label


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics, result, results, label = evaluate(args)
    thresholds = json.loads(Path(args.thresholds).read_text())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report.render(label, metrics, result, results, thresholds, args.judge))
    (out / "results.json").write_text(json.dumps({
        "label": label, "metrics": metrics, "gate_passed": result.passed, "gate_failures": result.failures,
        "cases": [{"id": r.case.id, "category": r.case.category, "answer": r.answer.text,
                   "retrieved": r.answer.retrieved, "citations": r.answer.citations,
                   "scores": r.scores, "judge": r.judge, "judge_reason": r.judge_reason} for r in results],
    }, indent=2))
    if args.write_baseline:
        Path(args.baseline).write_text(json.dumps({"label": label, "metrics": metrics}, indent=2) + "\n")

    print(f"{label}: {len(results)} cases, gate {'PASSED' if result.passed else 'FAILED'}. Report: {out / 'report.md'}")
    for failure in result.failures:
        print(f"  - {failure}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
