FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git unzip ca-certificates \
    make build-essential \
    autoconf automake libtool libtool-bin pkg-config m4 g++ \
    zlib1g-dev mecab libmecab-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir konlpy mecab-python3 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/haven-jeon/PyKoSpacing.git

COPY ./DPDT/mecab-ko-dic/ ./mecab-ko-dic/
RUN unzip ./mecab-ko-dic/matrix_def.zip -d ./mecab-ko-dic/ && \ 
    rm -f ./mecab-ko-dic/matrix_def.zip

COPY . .

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "main:app"]
