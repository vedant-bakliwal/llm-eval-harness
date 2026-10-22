# llm-eval-harness

Python 3.10+, no required dependencies. Evaluates a small RAG assistant (BM25 retrieval +
generator) over a golden set and gates releases on the scores.

- Tests: `pip install -e ".[dev]" && pytest -q`
- Eval: `python -m evalharness run` (offline). Real model: `--generator bedrock`; judge: `--judge bedrock|deepeval`.
- Re-baseline only on purpose: `python -m evalharness run --write-baseline`, and commit `evals/baseline.json` with a reason.

Rules for changes:
- Every new failure mode gets a golden case first, then the fix.
- Deterministic metrics stay dependency-free; anything needing an LLM lives in `judges/`.
- The handbook in `data/handbook` is invented. `vendors.md` contains a planted prompt injection on purpose.
- Tests must not call the network; use a fake client like `FakeBedrock` in `tests/test_end_to_end.py`.
