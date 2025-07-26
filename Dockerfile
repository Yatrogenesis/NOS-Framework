FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p /app/results /app/figures /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg

# Run tests to verify installation
RUN python -m pytest tests/ -v

# Default command
CMD ["python", "run_nos.py", "--config", "config/default.json"]