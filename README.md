# eval-demo

A demonstration project showcasing Python environment setup with a lightweight FastAPI server.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
python app.py
```

The server will start on `http://localhost:8000`

You can also run with uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

- `GET /` - Hello world message
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## Development

### Linting

This project uses Ruff for linting and formatting:

```bash
# Install dev dependencies
pip install -r requirements.txt ruff

# Run linting
ruff check .

# Run formatting
ruff format .
```

## Project Structure

```
eval-demo/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── pyproject.toml     # Project configuration
└── README.md          # This file
```
