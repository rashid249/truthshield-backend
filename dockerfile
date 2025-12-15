# -----------------------------
# 1. Base Python image
# -----------------------------
FROM python:3.10-slim

# Don't generate .pyc files and force unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -----------------------------
# 2. Working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# 3. System dependencies (if needed)
# -----------------------------
# For things like Pillow, requests, etc. this is enough.
# If later you use OpenCV or other libs, you might need extra packages.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 4. Install Python dependencies
# -----------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 5. Copy backend source code
# -----------------------------
COPY . .

# -----------------------------
# 6. Expose port 8000 inside the container
# -----------------------------
EXPOSE 8000

# -----------------------------
# 7. Start FastAPI with Uvicorn
# -----------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
