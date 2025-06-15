# ===== Stage 1: Build stage =====
FROM python:3.10-slim AS build-stage

WORKDIR /app

# 시스템 의존성 설치 (MeCab 및 빌드 도구 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git make build-essential \
    autoconf automake libtool pkg-config \
    libtool-bin m4 g++ \
    python3-pip unzip zlib1g-dev \
    mecab libmecab-dev mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

# mecab-ko-dic 전체 복사 (GitHub 버전 기준)
COPY ./DPDT/mecab-ko-dic/final/ ./mecab-ko-dic/

# automake 더미 파일 생성 (필요시)
RUN touch mecab-ko-dic/AUTHORS mecab-ko-dic/ChangeLog mecab-ko-dic/NEWS mecab-ko-dic/README

# 빌드 및 설치
RUN chmod +x mecab-ko-dic/autogen.sh && \
    cd mecab-ko-dic && ./autogen.sh && ./configure && make && make install

# ===== Stage 2: Runtime stage =====
FROM python:3.10-slim AS runtime

WORKDIR /app

# 1단계에서 빌드된 mecab 파일만 복사
COPY --from=build-stage /usr/local /usr/local

# 필수 Python 패키지 설치
RUN pip install --upgrade pip && \
    pip install konlpy && \
    pip install git+https://github.com/haven-jeon/PyKoSpacing.git

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# segment-anything 설치
RUN git clone https://github.com/facebookresearch/segment-anything.git && \
    cd segment-anything && \
    pip install -e .

# 애플리케이션 소스 복사
COPY . .

# 실행
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "main:app"]


# FROM python:3.10-slim

# WORKDIR /app

# # 시스템 의존성 설치 (MeCab 빌드에 필요한 도구 포함)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl git make build-essential \
#     autoconf automake libtool pkg-config \
#     libtool-bin m4 g++ \
#     python3-pip unzip zlib1g-dev \
#     && rm -rf /var/lib/apt/lists/*

# # MeCab-ko-dic 설치 (로컬 복사본 기반 빌드)
# COPY ./DPDT/mecab-ko-dic/ ./mecab-ko-dic/
# RUN ls -al ./mecab-ko-dic/

# # 🔧 automake 필수 더미 파일 생성
# RUN touch mecab-ko-dic/AUTHORS mecab-ko-dic/ChangeLog mecab-ko-dic/NEWS mecab-ko-dic/README

# # 🔧 matrix.def 생성
# RUN cd mecab-ko-dic/utils && \
#     g++ -o matrix-builder matrix_builder.cpp && \
#     ./matrix-builder > ../matrix.def
    
# RUN chmod +x mecab-ko-dic/autogen.sh
# RUN cd mecab-ko-dic && ./autogen.sh
# RUN cd mecab-ko-dic && ./configure
# RUN cd mecab-ko-dic && make
# RUN cd mecab-ko-dic && make install
# RUN cd .. && rm -rf mecab-ko-dic
    
# # konlpy + PyKoSpacing 설치
# RUN pip install --upgrade pip
# RUN pip install konlpy
# RUN pip3 install git+https://github.com/haven-jeon/PyKoSpacing.git

# # 나머지 종속성 설치 (sentence-transformers 등)
# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt

# # segment-anything 설치 (GitHub에서 clone 후 editable install)
# RUN git clone https://github.com/facebookresearch/segment-anything.git && \
#     cd segment-anything && \
#     pip install -e .

# # 앱 복사 및 실행
# COPY . .
# CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "main:app"]
