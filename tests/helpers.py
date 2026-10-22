from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "handbook"
CASES = ROOT / "evals" / "golden.jsonl"
THRESHOLDS = ROOT / "evals" / "thresholds.json"
BASELINE = ROOT / "evals" / "baseline.json"
