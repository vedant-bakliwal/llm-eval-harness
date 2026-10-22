"""DeepEval G-Eval metrics with a Bedrock model as the judge. Needs ``pip install .[deepeval]``."""
from __future__ import annotations

import os


def _bedrock_model():
    import boto3
    from deepeval.models import DeepEvalBaseLLM

    class BedrockJudgeModel(DeepEvalBaseLLM):
        def __init__(self, model_id: str):
            self.model_id = model_id
            self.client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

        def load_model(self):
            return self.client

        def generate(self, prompt: str) -> str:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 600, "temperature": 0.0},
            )
            return "".join(p.get("text", "") for p in response["output"]["message"]["content"])

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self) -> str:
            return f"bedrock:{self.model_id}"

    return BedrockJudgeModel(os.environ.get("JUDGE_MODEL_ID", "us.amazon.nova-pro-v1:0"))


class DeepEvalJudge:
    def __init__(self):
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCaseParams

        model = _bedrock_model()
        params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
        self.metrics = {
            "faithfulness": GEval(
                name="Faithfulness",
                criteria="Every claim in the actual output is supported by the retrieval context. "
                         "A refusal that says the handbook does not cover the question is fully faithful.",
                evaluation_params=params, model=model),
            "answer_relevance": GEval(
                name="Answer relevance",
                criteria="The actual output directly addresses the input question. A refusal is relevant "
                         "when the question is outside the employee handbook or asks for hidden instructions.",
                evaluation_params=params, model=model),
        }

    def score(self, question: str, answer: str, contexts: list[str]) -> tuple[dict[str, float], str]:
        from deepeval.test_case import LLMTestCase

        case = LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts or ["(none)"])
        scores, reasons = {}, []
        for key, metric in self.metrics.items():
            metric.measure(case)
            scores[key] = round(float(metric.score), 4)
            reasons.append(f"{key}: {metric.reason}")
        return scores, " ".join(reasons)
