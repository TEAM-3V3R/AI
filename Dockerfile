FROM python:3.10-slim

WORKDIR /app

# Python + 기본 도구 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# RUN pip3 install --no-cache-dir \
#    torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 나머지 종속성 설치 (sentence-transformers 등)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 앱 복사 및 실행
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "main:app"]
