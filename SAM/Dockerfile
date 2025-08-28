FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/sam_vit_b_01ec64.pth

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY segment-anything/ ./segment-anything/
RUN pip install --no-cache-dir ./segment-anything

RUN mkdir -p /app/models

COPY . .

CMD ["gunicorn", "-w", "1", "-t", "600", "-b", "0.0.0.0:5002", "sam:app"]
