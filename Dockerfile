FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치 (MeCab 빌드에 필요한 도구 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git make build-essential \
    autoconf automake libtool pkg-config \
    libtool-bin m4 g++ \
    python3-pip unzip zlib1g-dev mecab libmecab-dev \
    # libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --upgrade pip && \
    pip install --no-cache-dir konlpy mecab-python3

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir git+https://github.com/haven-jeon/PyKoSpacing.git
# RUN pip install --no-cache-dir git+https://github.com/CASIA-IVA-Lab/FastSAM.git

COPY ./DPDT/mecab-ko-dic/ ./mecab-ko-dic/
RUN unzip ./mecab-ko-dic/matrix_def.zip -d ./mecab-ko-dic/

COPY ./segment-anything/ ./segment-anything/
RUN pip install -e ./segment-anything

COPY . .

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "main:app"]
