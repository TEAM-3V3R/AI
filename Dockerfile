FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치 (MeCab 빌드에 필요한 도구 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git make build-essential \
    autoconf automake libtool pkg-config \
    libtool-bin m4 g++ \
    python3-pip unzip zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# MeCab-ko-dic 설치 (로컬 복사본 기반 빌드)
COPY ./DPDT/mecab-ko-dic/ ./mecab-ko-dic/
RUN ls -al ./mecab-ko-dic/

# 🔧 automake 필수 더미 파일 생성
RUN touch mecab-ko-dic/AUTHORS mecab-ko-dic/ChangeLog mecab-ko-dic/NEWS mecab-ko-dic/README

# 🔧 matrix.def 생성
RUN cd mecab-ko-dic/utils && \
    g++ -o matrix-builder matrix_builder.cpp && \
    ./matrix-builder > ../matrix.def
    
RUN chmod +x mecab-ko-dic/autogen.sh
RUN cd mecab-ko-dic && ./autogen.sh
RUN cd mecab-ko-dic && ./configure
RUN cd mecab-ko-dic && make
RUN cd mecab-ko-dic && make install
RUN cd .. && rm -rf mecab-ko-dic
    
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
