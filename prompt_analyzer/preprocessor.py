# prompt_analyzer/preprocessor.py

import os
# TF가 GPU를 못 보게 만들어서 libdevice 에러 회피
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re
import logging
import subprocess
from pathlib import Path
from pykospacing import Spacing
from konlpy.tag import Mecab

# ────────────────────────────────────────────────────────────────
# 환경 설정
# ────────────────────────────────────────────────────────────────

# 1) 환경변수로 mecabrc 위치 지정 (konlpy 내부에서 사용)
os.environ["MECABRC"] = "/etc/mecabrc"

# 2) 프로젝트 루트 및 데이터 경로
BASE_DIR = Path(__file__).resolve().parent.parent
STOPWORDS_PATH = BASE_DIR / "DPDT" / "data" / "stopwords.txt"

# 3) 띄어쓰기 교정기 (모델 로딩 비용이 크니 전역에서 한 번만)
spacing = Spacing()

# 4) MeCab 사전 경로 지정
mecab_dic =  "/app/mecab-ko-dic"

# 5) 형태소 분석기 초기화
tagger = Mecab(dicpath=mecab_dic)

# 6) 불용어 정의
#    STOP_POS: 제거할 품사 목록 (조사, 어미, 구두점, 접미사 등)
STOP_POS = {
    "JKS","JKC","JKO","JKB","JX","JC","JKG",
    "EF","EC","EP","ETN","ETM",
    "SF","SE","SP","SS","SO",
    "XSN","XSV"
}
#    WORD_STOP: 제거할 단어 목록 (data/stopwords.txt)
with STOPWORDS_PATH.open(encoding="utf8") as f:
    WORD_STOP = { w.strip() for w in f if w.strip() and not w.startswith("#") }

# ────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    1) spacing 모델로 띄어쓰기 보정 (실패 시 원문 유지)
    2) 한글·공백 외 문자 모두 제거
    3) 연속된 공백 하나로 합치고 양끝 strip
    """
    try:
        text = spacing(text)
    except Exception as e:
        logging.warning(f"[clean_text] spacing failed: {e}")
        # 보정 실패 시 원문 유지

    # 한글·공백만 남기기
    text = re.sub(r"[^가-힣\s]", " ", text)
    # 여러 공백 → 하나
    return re.sub(r"\s+", " ", text).strip()

def extract_morphs(text: str) -> list[tuple[str, str]]:
    """
    1) clean_text로 전처리
    2) MeCab 형태소 분석 → (단어, 품사) 리스트
    3) STOP_POS, WORD_STOP 조건으로 필터링
    """
    cleaned = clean_text(text)
    morphs = tagger.pos(cleaned)

    result = []
    for w, pos in morphs:
        if not w:
            continue
        if w in WORD_STOP:
            continue
        if pos in STOP_POS:
            continue
        result.append((w, pos))

    return result
