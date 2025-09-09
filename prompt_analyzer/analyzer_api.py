"""
분석 API
- POST /analyze : {"chatId": <int?>, "promptContents": [<str>, ...], "model_name"?, "centroids_path"?}
- GET  /healthz  : 헬스체크
- 기본 실행      : python -m AI.prompt_analyzer.analyzer_api 
- 로컬 셀프테스트: python -m AI.prompt_analyzer.analyzer_api --selftest
"""

import os
import json
import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- import ---
try:
    from AI.prompt_analyzer.fluency import compute_fluency
    from AI.prompt_analyzer.persistence import compute_persistence
except Exception:
    from prompt_analyzer.fluency import compute_fluency  # type: ignore
    from prompt_analyzer.persistence import compute_persistence  # type: ignore

# KoBERT
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
    model_name: str = "skt/kobert-base-v1",  
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
            "status": 200,
            "message": "분석이 성공적으로 완료되었습니다.",
            "results": [
                {
                    "creativity": round(creativity_score, 4),
                    "fluency": round(flu_score, 4),
                    "persistence": round(pers_score, 4),
                    "fluency_skc": {
                        "fluency_s": round(fS, 4),
                        "fluency_k": round(fK, 4),
                        "fluency_c": round(fC, 4),
                    },
                    "persistence_srf": {
                        "persistence_s": round(pS, 4),
                        "persistence_r": round(pR, 4),
                        "persistence_f": round(pF, 4)
                    }
                }
            ]
        }



    except Exception as e:
        print("분석 중 예외 발생:", e, flush=True)
        return {
            "error": str(e),
            "status": 500,
            "timestamp": datetime.datetime.now().isoformat(),
        }

# Flask 앱 (엔드포인트)
app = Flask(__name__)
CORS(app)

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200

@app.route("/analyze", methods=["POST"], strict_slashes=False)
def analyze_route():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON. Expect an object.",
            "status": 400
        }), 400

    chat_id = None
    texts = data.get("promptContents")
    if not isinstance(texts, list) or not all(isinstance(x, str) for x in texts):
        return jsonify({
            "error": "Invalid 'promptContents'. Must be a list of strings.",
            "status": 400
        }), 400
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return jsonify({
            "error": "No non-empty strings in 'promptContents'.",
            "status": 400
        }), 400
    
    model_name = data.get("model_name") or "skt/kobert-base-v1"
    centroids_path = data.get("centroids_path") or DEFAULT_CENTROIDS_PATH

    print(f"[API] analyze: n_texts={len(texts)}, model='{model_name}', centroids='{centroids_path}', chat_id={chat_id}", flush=True)

    result = analyze_from_api(texts, centroids_path=centroids_path, model_name=model_name)
    if chat_id is not None:
        result["chatId"] = chat_id
    result["accepted_key"] = "promptContents"

    code = result.get("status", 200)
    return jsonify(result), int(code)


# 메인: 기본은 서버 실행, --selftest 시 샘플 출력 후 종료
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
    ap.add_argument("--debug", action="store_true", default=True)
    ap.add_argument("--selftest", action="store_true", help="샘플 문장으로 로컬 분석만 수행하고 종료")
    ap.add_argument("--centroids", default=DEFAULT_CENTROIDS_PATH, help="override centroids path")
    ap.add_argument("--model", default="skt/kobert-base-v1")
    args = ap.parse_args()

    if args.selftest:
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
        out = analyze_from_api(samples, centroids_path=args.centroids, model_name=args.model)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)
