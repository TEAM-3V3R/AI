FROM python:3.10-slim

WORKDIR /app

# Python + 기본 도구 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    mecab libmecab-dev mecab-ipadic-utf8 mecab-config \
    curl git make build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install git+https://github.com/haven-jeon/PyKoSpacing.git

# 나머지 종속성 설치 (sentence-transformers 등)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# segment-anything 설치 (GitHub에서 clone 후 editable install)
RUN git clone https://github.com/facebookresearch/segment-anything.git && \
    cd segment-anything && \
    pip install -e .

# 앱 복사 및 실행
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "main:app"]
