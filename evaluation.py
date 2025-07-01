"""
LangChain evaluation module for testing LLM responses.
"""

from typing import Any, Dict, List

import requests
from fastapi import HTTPException
from langchain.evaluation import EvaluatorType, load_evaluator
from pydantic import BaseModel


class EvaluationCase(BaseModel):
    """Pydantic model for evaluation test cases."""

    prompt: str
    expected_output: str
    test_name: str


class LLMEvaluator:
    """LLM response evaluator using LangChain."""

    def __init__(self):
        """Initialize evaluators."""
        self.exact_match_evaluator = load_evaluator(EvaluatorType.EXACT_MATCH)
        self.string_distance_evaluator = load_evaluator(EvaluatorType.STRING_DISTANCE)

    def evaluate_response(
        self, actual_output: str, expected_output: str
    ) -> Dict[str, Any]:
        """
        Evaluate actual LLM output against expected output.

        Args:
            actual_output: The actual response from the LLM
            expected_output: The expected response

        Returns:
            Dictionary containing evaluation results
        """
        try:
            exact_result = self.exact_match_evaluator.evaluate_strings(
                prediction=actual_output, reference=expected_output
            )

            distance_result = self.string_distance_evaluator.evaluate_strings(
                prediction=actual_output, reference=expected_output
            )

            return {
                "exact_match_score": exact_result["score"],
                "string_distance_score": distance_result["score"],
                "exact_match": exact_result["score"] == 1.0,
                "similarity_threshold_met": distance_result["score"] <= 0.3,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

    def run_evaluation(
        self, test_cases: List[EvaluationCase], base_url: str = "http://localhost:8000"
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation on multiple test cases using actual HTTP requests.

        Args:
            test_cases: List of evaluation test cases
            base_url: Base URL for the API server

        Returns:
            List of evaluation results
        """
        results = []

        for case in test_cases:
            try:
                response = requests.post(
                    f"{base_url}/llm",
                    params={"prompt": case.prompt},
                    timeout=30
                )

                if response.status_code != 200:
                    results.append(
                        {
                            "test_name": case.test_name,
                            "prompt": case.prompt,
                            "expected_output": case.expected_output,
                            "actual_output": None,
                            "error": f"LLM API error: {response.status_code}",
                            "evaluation_results": None,
                        }
                    )
                    continue

                data = response.json()
                if "response" not in data:
                    results.append(
                        {
                            "test_name": case.test_name,
                            "prompt": case.prompt,
                            "expected_output": case.expected_output,
                            "actual_output": None,
                            "error": "LLM API response missing 'response' key",
                            "evaluation_results": None,
                        }
                    )
                    continue

                actual_output = data["response"]

                evaluation_results = self.evaluate_response(
                    actual_output, case.expected_output
                )

                results.append(
                    {
                        "test_name": case.test_name,
                        "prompt": case.prompt,
                        "expected_output": case.expected_output,
                        "actual_output": actual_output,
                        "error": None,
                        "evaluation_results": evaluation_results,
                    }
                )

            except requests.exceptions.RequestException as e:
                results.append(
                    {
                        "test_name": case.test_name,
                        "prompt": case.prompt,
                        "expected_output": case.expected_output,
                        "actual_output": None,
                        "error": f"HTTP request failed: {str(e)}",
                        "evaluation_results": None,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "test_name": case.test_name,
                        "prompt": case.prompt,
                        "expected_output": case.expected_output,
                        "actual_output": None,
                        "error": str(e),
                        "evaluation_results": None,
                    }
                )

        return results
