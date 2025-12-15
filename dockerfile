FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/hf
ENV HF_HOME=/tmp/hf

WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
