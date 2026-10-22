"""LLM-as-judge on Amazon Bedrock with a fixed rubric. Needs ``pip install .[bedrock]``."""
from __future__ import annotations

import json
import os
import re

RUBRIC = """You grade answers from a company-handbook assistant.
Score each criterion from 1 (bad) to 5 (excellent):
- faithfulness: every claim in the answer is supported by the context. A correct refusal scores 5.
- answer_relevance: the answer addresses the question that was asked. A refusal for an out-of-scope question scores 5.
Reply with only JSON: {"faithfulness": n, "answer_relevance": n, "reason": "one sentence"}"""


class BedrockJudge:
    def __init__(self, model_id: str | None = None, region: str | None = None, client=None):
        self.model_id = model_id or os.environ.get("JUDGE_MODEL_ID", "us.amazon.nova-pro-v1:0")
        if client is None:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))
        self.client = client

    def score(self, question: str, answer: str, contexts: list[str]) -> tuple[dict[str, float], str]:
        prompt = f"Question: {question}\n\nContext:\n" + "\n---\n".join(contexts or ["(none)"]) + f"\n\nAnswer: {answer}"
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": RUBRIC}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 200, "temperature": 0.0},
        )
        text = "".join(p.get("text", "") for p in response["output"]["message"]["content"])
        return parse(text)


def parse(text: str) -> tuple[dict[str, float], str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return JSON: {text[:200]}")
    data = json.loads(match.group(0))
    scores = {}
    for key in ("faithfulness", "answer_relevance"):
        value = float(data[key])
        if not 1 <= value <= 5:
            raise ValueError(f"Judge score out of range for {key}: {value}")
        scores[key] = round((value - 1) / 4, 4)  # 1..5 -> 0..1
    return scores, str(data.get("reason", ""))
