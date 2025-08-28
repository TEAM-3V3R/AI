FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/sam_vit_b_01ec64.pth \
    PORT=5002

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      flask gunicorn pillow requests opencv-python-headless numpy && \
    pip install --no-cache-dir "git+https://github.com/facebookresearch/segment-anything.git"

RUN mkdir -p /app/models

COPY . .

EXPOSE ${PORT}
CMD ["gunicorn", "-w", "1", "-t", "600", "-b", "0.0.0.0:${PORT}", "sam:app"]
