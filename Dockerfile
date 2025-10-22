# -------------------------------
# Base image
# -------------------------------
FROM python:3.11-slim

# -------------------------------
# Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Install system dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Copy requirements for caching
# -------------------------------
COPY requirements.txt .

# -------------------------------
# Install Python dependencies
# -------------------------------
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Copy application code
# -------------------------------
COPY . .

# -------------------------------
# Expose port (Railway uses $PORT)
# -------------------------------
EXPOSE 8000

# -------------------------------
# CMD: shell form for env var expansion
# -------------------------------
CMD sh -c "uvicorn run:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"
