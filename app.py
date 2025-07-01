"""
Simple FastAPI server for demonstration purposes.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from evaluation import EvaluationCase, LLMEvaluator

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="eval-demo", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
templates = Jinja2Templates(directory="templates")


class EvaluationRequest(BaseModel):
    """Request model for evaluation endpoint."""

    test_cases: List[Dict[str, str]]


@app.get("/")
def hello_world():
    """Basic hello world endpoint."""
    return {"message": "Hello, World!", "status": "success", "api_version": "1.0"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "eval-demo-api"}


@app.post("/llm")
def call_openai_llm(prompt: str = "Hello") -> Dict[str, Any]:
    """
    Make an LLM call to OpenAI API with the given prompt.

    Args:
        prompt: The prompt to send to the LLM (default: "Hello")

    Returns:
        Dict containing the LLM response or error information
    """
    try:
        url = "https://api.openai.com/v1/chat/completions"

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.7,
        }

        json_data = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=json_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_get_openai_api_key()}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"]
            return {
                "status": "success",
                "prompt": prompt,
                "response": content,
                "model": data["model"],
            }
        else:
            return {
                "status": "error",
                "message": "No response from OpenAI API",
                "prompt": prompt,
            }

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else "No error details"
        raise HTTPException(
            status_code=e.code,
            detail=f"OpenAI API error: {e.reason}. Details: {error_body}",
        )
    except urllib.error.URLError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {e.reason}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON decode error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


def _get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment variable.

    Returns:
        The API key string

    Raises:
        HTTPException: If API key is not found
    """
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="OPENAI_API_KEY environment variable not set"
        )
    return api_key


@app.get("/evaluate-ui", response_class=HTMLResponse)
def evaluate_ui(request: Request):
    """Serve the evaluation UI."""
    return templates.TemplateResponse("evaluate.html", {"request": request})


@app.post("/evaluate")
@limiter.limit("10/hour")
def evaluate_llm_responses(
    request: Request, evaluation_request: EvaluationRequest
) -> Dict[str, Any]:
    """
    Evaluate LLM responses against expected outputs.
    Rate limited to 10 requests per hour per IP address.

    Args:
        request: FastAPI request object (for rate limiting)
        evaluation_request: Evaluation request containing test cases

    Returns:
        Dictionary containing evaluation results
    """
    try:
        test_cases = []
        for case_data in evaluation_request.test_cases:
            test_cases.append(
                EvaluationCase(
                    prompt=case_data["prompt"],
                    expected_output=case_data["expected_output"],
                    test_name=case_data.get("test_name", "unnamed_test"),
                )
            )

        evaluator = LLMEvaluator()

        base_url = "http://localhost:8000"
        results = evaluator.run_evaluation(test_cases, base_url)

        return {"status": "success", "total_tests": len(test_cases), "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
