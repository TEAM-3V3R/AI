FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치 (MeCab 빌드에 필요한 도구 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git make build-essential \
    autoconf automake libtool pkg-config \
    libtool-bin m4 g++ \
    python3-pip unzip zlib1g-dev mecab \
    && rm -rf /var/lib/apt/lists/*
    
# 종속성 설치
RUN echo "[STEP 2] pip 업그레이드 및 konlpy"
RUN pip install --upgrade pip
RUN pip install konlpy

# 🔥 MeCab Python 바인딩 설치 (이게 빠져있었음)
RUN pip install python-mecab-ko

RUN echo "[STEP 3] requirements.txt 설치"
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


RUN echo "[STEP 4] PyKoSpacing 설치"
RUN pip3 install git+https://github.com/haven-jeon/PyKoSpacing.git

# MeCab-ko-dic 복사 및 matrix.def 압축 해제
RUN echo "[STEP 5] MeCab-ko-dic 복사"
COPY ./DPDT/mecab-ko-dic/ ./mecab-ko-dic/
RUN unzip ./mecab-ko-dic/matrix_def.zip -d ./mecab-ko-dic/

# segment-anything 설치 (GitHub에서 clone 후 editable install)
RUN echo "[STEP 6] segment-anything 복사 및 설치"
COPY ./segment-anything ./segment-anything
RUN cd segment-anything && \
    pip install -e .

# 앱 복사 및 실행
RUN echo "[STEP 7] 앱 전체 복사"
COPY . .
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "main:app"]
