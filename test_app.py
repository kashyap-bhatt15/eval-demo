"""
Unit tests for the FastAPI application.
"""

import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app import _get_openai_api_key, app


class TestApp(unittest.TestCase):
    """Test cases for the FastAPI application."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_hello_world(self):
        """Test the hello world endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["message"], "Hello, World!")
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["api_version"], "1.0")

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "eval-demo-api")

    @patch("os.getenv")
    @patch("app.urllib.request.urlopen")
    def test_llm_endpoint_success(self, mock_urlopen, mock_getenv):
        """Test successful LLM API call."""
        mock_getenv.return_value = "test-api-key"

        mock_response_data = {
            "choices": [{"message": {"content": "Hello! How can I help you today?"}}]
        }

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        response = self.client.post("/llm?prompt=Hello")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["prompt"], "Hello")
        self.assertEqual(data["response"], "Hello! How can I help you today?")
        self.assertEqual(data["model"], "gpt-3.5-turbo")

    @patch("os.getenv")
    @patch("app.urllib.request.urlopen")
    def test_llm_endpoint_default_prompt(self, mock_urlopen, mock_getenv):
        """Test LLM endpoint with default prompt."""
        mock_getenv.return_value = "test-api-key"

        mock_response_data = {
            "choices": [{"message": {"content": "Hello! How can I help you today?"}}]
        }

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        response = self.client.post("/llm")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["prompt"], "Hello")  # Default prompt

    @patch("os.getenv")
    def test_llm_endpoint_missing_api_key(self, mock_getenv):
        """Test LLM endpoint when API key is missing."""
        mock_getenv.return_value = None

        response = self.client.post("/llm?prompt=Hello")
        self.assertEqual(response.status_code, 500)

        data = response.json()
        self.assertIn("OPENAI_API_KEY environment variable not set", data["detail"])

    @patch("os.getenv")
    @patch("app.urllib.request.urlopen")
    def test_llm_endpoint_http_error(self, mock_urlopen, mock_getenv):
        """Test LLM endpoint when OpenAI API returns HTTP error."""
        mock_getenv.return_value = "test-api-key"

        import io

        error_response = io.BytesIO(b'{"error": "Unauthorized"}')
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            401,
            "Unauthorized",
            None,
            error_response,
        )

        response = self.client.post("/llm?prompt=Hello")
        self.assertEqual(response.status_code, 401)

        data = response.json()
        self.assertIn("OpenAI API error", data["detail"])

    @patch("os.getenv")
    @patch("app.urllib.request.urlopen")
    def test_llm_endpoint_network_error(self, mock_urlopen, mock_getenv):
        """Test LLM endpoint when network error occurs."""
        mock_getenv.return_value = "test-api-key"

        error = urllib.error.URLError("Network unreachable")
        mock_urlopen.side_effect = error

        response = self.client.post("/llm?prompt=Hello")
        self.assertEqual(response.status_code, 500)

        data = response.json()
        self.assertIn("Network error", data["detail"])

    @patch("os.getenv")
    @patch("app.urllib.request.urlopen")
    def test_llm_endpoint_json_decode_error(self, mock_urlopen, mock_getenv):
        """Test LLM endpoint when JSON decode error occurs."""
        mock_getenv.return_value = "test-api-key"

        mock_response = MagicMock()
        mock_response.read.return_value = b"invalid json"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        response = self.client.post("/llm?prompt=Hello")
        self.assertEqual(response.status_code, 500)

        data = response.json()
        self.assertIn("JSON decode error", data["detail"])

    @patch("os.getenv")
    @patch("app.urllib.request.urlopen")
    def test_llm_endpoint_no_choices_in_response(self, mock_urlopen, mock_getenv):
        """Test LLM endpoint when OpenAI response has no choices."""
        mock_getenv.return_value = "test-api-key"

        mock_response_data = {"usage": {"total_tokens": 10}}

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        response = self.client.post("/llm?prompt=Hello")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["message"], "No response from OpenAI API")
        self.assertEqual(data["prompt"], "Hello")

    @patch("os.getenv")
    def test_get_openai_api_key_success(self, mock_getenv):
        """Test successful API key retrieval."""
        mock_getenv.return_value = "test-api-key"
        api_key = _get_openai_api_key()
        self.assertEqual(api_key, "test-api-key")

    @patch("os.getenv")
    def test_get_openai_api_key_missing(self, mock_getenv):
        """Test API key retrieval when key is missing."""
        mock_getenv.return_value = None

        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as context:
            _get_openai_api_key()

        self.assertIn(
            "OPENAI_API_KEY environment variable not set", str(context.exception.detail)
        )


if __name__ == "__main__":
    unittest.main()
