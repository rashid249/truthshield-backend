# ----------------------------
# Base image (lightweight)
# ----------------------------
FROM python:3.10-slim

# ----------------------------
# Environment variables
# ----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ----------------------------
# Working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Copy requirements first (cache-friendly)
# ----------------------------
COPY requirements.txt .

# ----------------------------
# Install Python dependencies
# IMPORTANT: Use PyTorch CPU index
# ----------------------------
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ----------------------------
# Copy application code
# ----------------------------
COPY . .

# ----------------------------
# Expose FastAPI port
# ----------------------------
EXPOSE 8000

# ----------------------------
# Start FastAPI app
# ----------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
