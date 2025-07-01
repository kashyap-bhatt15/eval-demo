"""
LangChain evaluation test cases for the FastAPI LLM endpoint.
"""

import unittest
from unittest.mock import MagicMock, patch

from evaluation import EvaluationCase, LLMEvaluator


class TestLangChainEvaluations(unittest.TestCase):
    """Test cases for LangChain evaluations of LLM responses."""

    def setUp(self):
        """Set up evaluators."""
        self.evaluator = LLMEvaluator()
        self.base_url = "http://localhost:8000"

    def test_evaluation_case_1_greeting(self):
        """Test Case 1: Greeting prompt evaluation."""
        expected_output = (
            "Hello! I'm doing well, thank you for asking. How can I help you today?"
        )

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": expected_output}
            mock_post.return_value = mock_response

            evaluation_results = self.evaluator.evaluate_response(
                expected_output, expected_output
            )

            self.assertTrue(
                evaluation_results["exact_match"], "Exact match should be true"
            )
            self.assertEqual(
                evaluation_results["exact_match_score"],
                1.0,
                "Exact match score should be 1.0",
            )
            self.assertLessEqual(
                evaluation_results["string_distance_score"],
                0.1,
                "String distance should be very low",
            )

    def test_evaluation_case_2_question_answering(self):
        """Test Case 2: Question answering prompt evaluation."""
        expected_output = "The capital of France is Paris."

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": expected_output}
            mock_post.return_value = mock_response

            evaluation_results = self.evaluator.evaluate_response(
                expected_output, expected_output
            )

            self.assertTrue(
                evaluation_results["exact_match"],
                "Exact match should be true for factual answer",
            )
            self.assertEqual(
                evaluation_results["exact_match_score"],
                1.0,
                "Exact match score should be 1.0",
            )

    def test_evaluation_case_3_creative_writing(self):
        """Test Case 3: Creative writing prompt evaluation."""
        expected_output = (
            "Waves crash upon the sandy shore,\n"
            "Blue depths hold secrets and much more.\n"
            "The ocean vast, both calm and wild,\n"
            "Nature's beauty, unreconciled."
        )

        mock_actual_output = (
            "The ocean blue stretches far and wide,\n"
            "With rolling waves and changing tide.\n"
            "A peaceful scene of endless sea,\n"
            "Where dolphins play so wild and free."
        )

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": mock_actual_output}
            mock_post.return_value = mock_response

            evaluation_results = self.evaluator.evaluate_response(
                mock_actual_output, expected_output
            )

            self.assertFalse(
                evaluation_results["exact_match"],
                "Creative writing should not match exactly",
            )
            self.assertIsNotNone(
                evaluation_results["string_distance_score"],
                "String distance should be calculated",
            )

    def test_evaluation_batch_processing(self):
        """Test batch evaluation of multiple test cases."""
        test_cases = [
            EvaluationCase(
                prompt="Hello, how are you?",
                expected_output=(
                    "Hello! I'm doing well, thank you for asking. "
                    "How can I help you today?"
                ),
                test_name="greeting",
            ),
            EvaluationCase(
                prompt="What is the capital of France?",
                expected_output="The capital of France is Paris.",
                test_name="factual_qa",
            ),
            EvaluationCase(
                prompt="Write a short poem about the ocean.",
                expected_output=(
                    "Waves crash upon the sandy shore,\n"
                    "Blue depths hold secrets and much more.\n"
                    "The ocean vast, both calm and wild,\n"
                    "Nature's beauty, unreconciled."
                ),
                test_name="creative_writing",
            ),
        ]

        with patch("requests.post") as mock_post:
            def mock_post_side_effect(url, **kwargs):
                if "/llm" in url:
                    prompt = kwargs.get("params", {}).get("prompt", "")

                    if "Hello" in prompt:
                        content = (
                            "Hello! I'm doing well, thank you for asking. "
                            "How can I help you today?"
                        )
                    elif "capital of France" in prompt:
                        content = "The capital of France is Paris."
                    else:
                        content = (
                            "The ocean blue stretches far and wide,\n"
                            "With rolling waves and changing tide.\n"
                            "A peaceful scene of endless sea,\n"
                            "Where dolphins play so wild and free."
                        )

                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"response": content}
                    return mock_response

                return MagicMock()

            mock_post.side_effect = mock_post_side_effect
            results = self.evaluator.run_evaluation(test_cases, self.base_url)

            self.assertEqual(len(results), 3, "Should have 3 evaluation results")

            for result in results:
                self.assertIsNotNone(result["test_name"], "Test name should be present")
                self.assertIsNotNone(result["prompt"], "Prompt should be present")
                self.assertIsNotNone(
                    result["expected_output"], "Expected output should be present"
                )
                self.assertIsNotNone(
                    result["actual_output"], "Actual output should be present"
                )
                self.assertIsNone(result["error"], "No errors should occur")
                self.assertIsNotNone(
                    result["evaluation_results"], "Evaluation results should be present"
                )


if __name__ == "__main__":
    unittest.main()
