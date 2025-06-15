FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git make build-essential \
    autoconf automake libtool pkg-config \
    python3-pip ca-certificates file locales unzip \
    && rm -rf /var/lib/apt/lists/*

# MeCab 설치 (정식 tar.gz 버전 사용)
RUN curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz && \
    tar zxfv mecab-0.996-ko-0.9.2.tar.gz && \
    cd mecab-0.996-ko-0.9.2 && \
    ./configure && make && make install && \
    cd .. && rm -rf mecab-0.996-ko-0.9.2*

# MeCab-ko-dic 설치
RUN curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz && \
    tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz && \
    cd mecab-ko-dic-2.1.1-20180720 && \
    ./configure --prefix=/usr/local --with-dicdir=/usr/local/lib/mecab/dic/mecab-ko-dic && \
    make && make install && \
    cd .. && rm -rf mecab-ko-dic*
    
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
