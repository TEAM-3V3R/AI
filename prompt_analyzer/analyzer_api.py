"""
분석 API
- POST /analyze : {"chatId": <int?>, "promptContents": [<str>, ...], "model_name"?, "centroids_path"?}
- GET  /healthz  : 헬스체크
"""

import os
import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from prompt_analyzer.fluency import compute_fluency  
from prompt_analyzer.persistence import compute_persistence  

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

# 기본 센트로이드 경로
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
    return "DPDT/models/kmeans_k100/centroids.npy"

DEFAULT_CENTROIDS_PATH = _default_centroids_path()

_TOKENIZER = None
_MODEL = None
_DEVICE = None
_CENTROIDS = None          
_CENTROIDS_TAG = None      

def _ensure_resources(model_name: str, centroids_path: str) -> None:
    global _TOKENIZER, _MODEL, _DEVICE, _CENTROIDS, _CENTROIDS_TAG

    if _TOKENIZER is None or _MODEL is None:
        print(f"[RES] loading model='{model_name}'", flush=True)
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        _MODEL = AutoModel.from_pretrained(model_name).eval()
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL.to(_DEVICE)
        print(f"[RES] device={_DEVICE}", flush=True)

    cp = Path(centroids_path)
    if not cp.exists():
        cands = "\n  - " + "\n  - ".join(str(p) for p in _DEFAULT_CENTROIDS_CANDIDATES)
        raise FileNotFoundError(f"centroids not found: {cp}\nTried candidates:{cands}")
    
    tag = cp.resolve().as_posix()
    if (_CENTROIDS is None) or (_CENTROIDS_TAG != tag):
        print(f"[RES] loading centroids='{tag}'", flush=True)
        arr = np.load(cp)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        _CENTROIDS = arr
        _CENTROIDS_TAG = tag
        print(f"[RES] centroids shape={_CENTROIDS.shape}, dtype={_CENTROIDS.dtype}", flush=True)

def _call_metric(func, texts: List[str], centroids_path: str, model_name: str) -> Tuple[float, float, float, float]:
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
        val = func(texts, centroids_path, model_name)

    if isinstance(val, tuple) and len(val) >= 4:
        score, S, K, C = val[:4]
        return float(score), float(S), float(K), float(C)
    return float(val), 0.0, 0.0, 0.0

def analyze_from_api(
    texts: List[str],
    centroids_path: str = DEFAULT_CENTROIDS_PATH,
    model_name: str = "klue/bert-base",  
) -> Dict[str, Any]:
    print("analyze_from_api 진입", flush=True)

    if not texts:
        print("입력 문장 없음", flush=True)
        return {
            "error": "분석할 문장이 없습니다.",
            "status": 400,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    try:
        _ensure_resources(model_name, centroids_path)

        print("compute_fluency 시작", flush=True)
        flu_score, fS, fK, fC = _call_metric(compute_fluency, texts, centroids_path, model_name)
        print("compute_fluency 완료:", flu_score, flush=True)

        print("compute_persistence 시작", flush=True)
        pers_score, pS, pR, pF = _call_metric(compute_persistence, texts, centroids_path, model_name)
        print("compute_persistence 완료:", pers_score, flush=True)

        creativity_score = (flu_score * 0.5 + pers_score * 0.5)
        print("최종 점수:", creativity_score, flush=True)

        return {
            "fluency": round(flu_score, 4),
            "persistence": round(pers_score, 4),
            "creativity": round(creativity_score, 4),
            "fluencySkc": {
                "fluency_s": round(fS, 4),
                "fluency_k": round(fK, 4),
                "fluency_c": round(fC, 4),
            },
            "persistenceSrf": {
                "persistence_s": round(pS, 4),
                "persistence_r": round(pR, 4),
                "persistence_f": round(pF, 4)
            }
        }




    except Exception as e:
        print("분석 중 예외 발생:", e, flush=True)
        return {
            "error": str(e),
            "status": 500,
            "timestamp": datetime.datetime.now().isoformat(),
        }