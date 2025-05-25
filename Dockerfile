FROM openjdk:11-jre-slim

# Python + 기본 도구 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements 설치 전 torch를 먼저 CUDA 12.1 버전으로 명시 설치
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 나머지 종속성 설치 (sentence-transformers 등)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 앱 복사 및 실행
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "main:app"]