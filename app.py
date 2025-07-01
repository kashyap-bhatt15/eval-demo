"""
Simple FastAPI server for demonstration purposes.
"""

import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any
from fastapi import FastAPI, HTTPException

app = FastAPI(title="eval-demo", version="1.0")


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
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        json_data = json.dumps(data).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=json_data,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {_get_openai_api_key()}'
            },
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            response_data = json.loads(response.read().decode('utf-8'))
            
        if 'choices' in response_data and len(response_data['choices']) > 0:
            content = response_data['choices'][0]['message']['content']
            return {
                "status": "success",
                "prompt": prompt,
                "response": content,
                "model": data["model"]
            }
        else:
            return {
                "status": "error",
                "message": "No response from OpenAI API",
                "prompt": prompt
            }
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else "No error details"
        raise HTTPException(
            status_code=e.code,
            detail=f"OpenAI API error: {e.reason}. Details: {error_body}"
        )
    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Network error: {e.reason}"
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"JSON decode error: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


def _get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment variable.
    
    Returns:
        The API key string
        
    Raises:
        HTTPException: If API key is not found
    """
    import os
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable not set"
        )
    return api_key


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
