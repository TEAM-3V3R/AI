import os
import json
import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

# --- 프로젝트 내 import (AI.접두어/무접두어 둘 다 지원) ---
try:
    from AI.prompt_analyzer.fluency import compute_fluency
    from AI.prompt_analyzer.persistence import compute_persistence
except Exception:
    from prompt_analyzer.fluency import compute_fluency  # type: ignore
    from prompt_analyzer.persistence import compute_persistence  # type: ignore

from transformers import AutoTokenizer, AutoModel

# KoBERT: fast 미지원 → 일관 설정
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

# ─────────────────────────────────────────────────────────────
# 기본 센트로이드 경로 추정
# ─────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CENTROIDS_CANDIDATES = [
    _THIS_DIR / "DPDT" / "models" / "kmeans_k100" / "centroids.npy",
    _THIS_DIR.parent / "DPDT" / "models" / "kmeans_k100" / "centroids.npy",
    Path("AI/DPDT/models/kmeans_k100/centroids.npy"),
    Path("DPDT/models/kmeans_k100/centroids.npy"),
]

def _default_centroids_path() -> str:
    for p in _DEFAULT_CENTROIDS_CANDIDATES:
        if Path(p).exists():
            return str(p)
    # 최후의 폴백(거의 사용되지 않음)
    return "DPDT/data/centroids.json"

DEFAULT_CENTROIDS_PATH = _default_centroids_path()

# ─────────────────────────────────────────────────────────────
# 전역 리소스 (lazy)
# ─────────────────────────────────────────────────────────────
_TOKENIZER = None
_MODEL = None
_DEVICE = None
_CENTROIDS = None          # np.ndarray(float32) (K, H)
_CENTROIDS_TAG = None      # 경로 해시

def _ensure_resources(model_name: str, centroids_path: str) -> None:
    """전역 리소스 1회 로드 & 재사용"""
    global _TOKENIZER, _MODEL, _DEVICE, _CENTROIDS, _CENTROIDS_TAG

    if _TOKENIZER is None or _MODEL is None:
        # KoBERT는 SentencePiece → use_fast=False
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        _MODEL = AutoModel.from_pretrained(model_name).eval()
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL.to(_DEVICE)

    cp = Path(centroids_path)
    tag = (cp.resolve().as_posix() if cp.exists() else str(cp))
    if (_CENTROIDS is None) or (_CENTROIDS_TAG != tag):
        if not cp.exists():
            raise FileNotFoundError(f"centroids not found: {cp}")
        arr = np.load(cp)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        _CENTROIDS = arr
        _CENTROIDS_TAG = tag

def _call_metric(func, texts, centroids_path: str, model_name: str) -> Tuple[float, float, float, float]:
    """
    compute_fluency / compute_persistence 호출 호환 래퍼
    - 신버전: (score, S, K, C) 반환
    - 구버전: score(float)만 반환
    - resources 인자를 지원하면 전달(속도↑), 아니면 일반 호출
    """
    # 우선 resources와 함께 호출 시도
    try:
        val = func(
            texts,
            centroids_path,
            model_name,
            resources={
                "tokenizer": _TOKENIZER,
                "model": _MODEL,
                "device": _DEVICE,
                "centroids": _CENTROIDS,
            },
        )
    except TypeError:
        # 구버전 시그니처로 재시도
        val = func(texts, centroids_path, model_name)

    # 반환값 정규화
    if isinstance(val, tuple) and len(val) >= 4:
        score, S, K, C = val[:4]
        return float(score), float(S), float(K), float(C)
    else:
        return float(val), 0.0, 0.0, 0.0

# ─────────────────────────────────────────────────────────────
# 외부 공개 단일 API
# ─────────────────────────────────────────────────────────────
def analyze_from_api(
    texts,
    centroids_path: str = DEFAULT_CENTROIDS_PATH,
    model_name: str = "skt/kobert-base-v1",
) -> Dict[str, Any]:
    print("📥 analyze_from_api 진입", flush=True)

    if not texts:
        print("⚠️ 입력 문장 없음", flush=True)
        return {
            "error": "분석할 문장이 없습니다.",
            "status": 400,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    try:
        _ensure_resources(model_name, centroids_path)

        print("🧠 compute_fluency 시작", flush=True)
        flu_score, fS, fK, fC = _call_metric(compute_fluency, texts, centroids_path, model_name)
        print("✅ compute_fluency 완료:", flu_score, flush=True)

        print("🧠 compute_persistence 시작", flush=True)
        pers_score, pS, pK, pC = _call_metric(compute_persistence, texts, centroids_path, model_name)
        print("✅ compute_persistence 완료:", pers_score, flush=True)

        creativity_score = (flu_score * 0.5 + pers_score * 0.5)
        print("🎯 최종 점수:", creativity_score, flush=True)

        return {
            "fluency": round(flu_score, 4),
            "persistence": round(pers_score, 4),
            "creativity": round(creativity_score, 4),
            "message": "분석이 성공적으로 완료되었습니다.",
            "status": 200,
            "timestamp": datetime.datetime.now().isoformat(),
            "detail": {
                "fluency_SKC": {"fluency_S": round(fS, 4), "fluency_K": round(fK, 4), "fluency_C": round(fC, 4)},
                "persistence_SKC": {"persistence_S": round(pS, 4), "persistence_R": round(pK, 4), "persistence_F": round(pC, 4)},
            },
        }

    except Exception as e:
        print("❌ 분석 중 예외 발생:", e, flush=True)
        return {
            "error": str(e),
            "status": 500,
            "timestamp": datetime.datetime.now().isoformat(),
        }

# ─────────────────────────────────────────────────────────────
# 로컬 셀프 테스트
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "안개 낀 숲길을 홀로 걷는 사람",
        "강아지가 뛰노는 푸른 들판",
        "도시의 밤거리를 달리는 자동차",
        "아이들이 공원에서 뛰어노는 장면",
        "바닷가에서 서핑을 즐기는 청년",
        "책상 위에 펼쳐진 고서와 만년필",
        "하늘 높이 떠 있는 열기구",
        "밤하늘을 수놓는 불꽃놀이",
        "유리창 너머로 비가 내리는 풍경",
        "산 정상에서 일출을 바라보는 등산객",
    ]
    out = analyze_from_api(samples, centroids_path=DEFAULT_CENTROIDS_PATH, model_name="skt/kobert-base-v1")
    print(json.dumps(out, ensure_ascii=False, indent=2))
