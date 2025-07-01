"""
Simple FastAPI server for demonstration purposes.
"""

from fastapi import FastAPI

app = FastAPI(title="eval-demo", version="1.0")


@app.get("/")
def hello_world():
    """Basic hello world endpoint."""
    return {"message": "Hello, World!", "status": "success", "api_version": "1.0"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "eval-demo-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
