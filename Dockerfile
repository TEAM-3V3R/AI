FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치 (MeCab-ko 빌드용 도구 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git make build-essential \
    autoconf automake libtool pkg-config \
    python3-pip unzip zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# MeCab-ko 빌드 (GitHub 미러 사용)
RUN git clone https://github.com/jonghwanhyeon/mecab-ko.git && \
    cd mecab-ko && ./autogen.sh && ./configure && make && make install && \
    cd .. && rm -rf mecab-ko

# MeCab-ko-dic 빌드 (GitHub 미러 사용)
RUN git clone https://github.com/jonghwanhyeon/mecab-ko-dic.git && \
    cd mecab-ko-dic && ./autogen.sh && ./configure && make && make install && \
    cd .. && rm -rf mecab-ko-dic
    
# konlpy + PyKoSpacing 설치
RUN pip install --upgrade pip
RUN pip install konlpy
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
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "main:app"]
