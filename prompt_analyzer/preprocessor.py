import os
import re
import logging
from pathlib import Path
from typing import List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 0) 띄어쓰기 교정기 (옵션)
try:
    from pykospacing import Spacing
    spacing = Spacing()
except Exception:
    spacing = None
    logging.warning("[preprocessor] pykospacing 미설치 → 띄어쓰기 교정 건너뜀")

# 1) 경로 설정
BASE_DIR: Path = Path(__file__).resolve().parents[1]
STOPWORDS_PATH: Path = BASE_DIR / "DPDT" / "data" / "stopwords.txt"
MECAB_DIC_DIR: Path = (BASE_DIR / "DPDT" / "mecab-ko-dic").resolve()

# 2) 형태소 분석기 로딩 (MeCab → 실패 시 Okt)
TAGGER_NAME = None
tagger = None

try:
    from konlpy.tag import Mecab

    if MECAB_DIC_DIR.exists():
        dicpath = str(MECAB_DIC_DIR).replace("\\", "/") 
        tagger = Mecab(dicpath=dicpath)
        TAGGER_NAME = f"mecab@{dicpath}"
    else:
        tagger = Mecab()
        TAGGER_NAME = "mecab(default)"

except Exception as e:
    logging.warning(f"[preprocessor] Mecab 초기화 실패: {e} → Okt로 폴백")
    try:
        from konlpy.tag import Okt
        tagger = Okt()
        TAGGER_NAME = "okt"
    except Exception as e2:
        raise RuntimeError(
            "형태소 분석기를 초기화할 수 없습니다. (Mecab/Okt 모두 실패)"
        ) from e2

# 3) 불용어/필터
STOP_POS = {
    "JKS","JKC","JKO","JKB","JX","JC","JKG",   # 조사
    "EF","EC","EP","ETN","ETM",                # 어미
    "SF","SE","SP","SS","SO",                  # 기호
    "XSN","XSV"                                # 접미사/접속
}

if STOPWORDS_PATH.exists():
    with STOPWORDS_PATH.open(encoding="utf-8") as f:
        WORD_STOP = {w.strip() for w in f if w.strip() and not w.startswith("#")}
else:
    WORD_STOP = set()

# 4) 텍스트 정리
def clean_text(text: str) -> str:
    """
    1) 띄어쓰기 보정(가능 시)
    2) 한글/공백만 유지
    3) 연속 공백 정리
    """
    if spacing:
        try:
            text = spacing(text)
        except Exception as e:
            logging.warning(f"[clean_text] spacing failed: {e}")  

    text = re.sub(r"[^가-힣\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# 5) 형태소 분석 + 필터링
def _pos_mecab(s: str) -> List[Tuple[str, str]]:
    return tagger.pos(s)

def _pos_okt(s: str) -> List[Tuple[str, str]]:
    return tagger.pos(s)

COMMON_VERBS = {"하다", "되다", "있다", "없다", "같다", "이다"}

def extract_morphs(text: str):
    s = clean_text(text)
    pairs = tagger.pos(s)

    result = []
    for w, pos in pairs:
        if not w:
            continue
        if len(w) == 1:   
            continue
        if w in WORD_STOP:
            continue
        if pos in STOP_POS:
            continue
        if w in COMMON_VERBS: 
            continue
        result.append((w, pos))
    return result


# 6) 테스트
if __name__ == "__main__":
    demo = "가디건을입고무표정하게정면을바라보는청년 입니다."
    print(f"[tagger={TAGGER_NAME}]")
    print("[raw    ]", demo)
    print("[cleaned]", clean_text(demo))
    print("[morphs ]", extract_morphs(demo))
