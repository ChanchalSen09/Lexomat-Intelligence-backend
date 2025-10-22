FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages with no cache to reduce size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY .env* ./

# Expose port (Railway uses $PORT)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start command
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

