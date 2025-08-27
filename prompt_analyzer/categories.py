# -*- coding: utf-8 -*-
# AI/prompt_analyzer/categories.py
import os
import json
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from flask import Blueprint, request, jsonify
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel

from AI.prompt_analyzer.preprocessor import extract_morphs

# ─────────────────────────────────────────
# 환경/로그
# ─────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.getLogger().setLevel(logging.INFO)

categories_bp = Blueprint("category", __name__, url_prefix="/category")

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
AI_DIR = Path(__file__).resolve().parents[1]        # .../AI
DPDT_DIR = AI_DIR / "DPDT"
DATA_DIR = DPDT_DIR / "data"
MODELS_DIR = DPDT_DIR / "models"

CENTROIDS_DEFAULT = MODELS_DIR / "kmeans_k100" / "centroids.npy"
RUNTIME_JSON      = DATA_DIR / "runtime_categories.json"  # 권장
SUPER_MAP_JSON    = DATA_DIR / "super_map.json"           # 폴백
SUPER_NAMES_JSON  = DATA_DIR / "super_names.json"         # 폴백

# ─────────────────────────────────────────
# 전역 캐시
# ─────────────────────────────────────────
_tokenizer = None
_model = None
_device = None
_kmeans = None
_centroid_to_super = None   # list[int] 길이=K  (또는 dict[str->int]를 list로 정규화)
_super_names = None         # dict[str->name]

# 품사 허용( MeCab + Okt 동시 지원 )
POS_ALLOWED = {
    # MeCab
    "NNG", "NNP", "NP", "VA",
    # Okt
    "Noun", "Adjective",
}

# ─────────────────────────────────────────
# 리소스 로드
# ─────────────────────────────────────────
def _load_mapping():
    """runtime_categories.json이 있으면 우선 사용.
       없으면 super_map.json(+super_names.json) 조합으로 폴백."""
    global _centroid_to_super, _super_names

    if RUNTIME_JSON.exists():
        cfg = json.loads(RUNTIME_JSON.read_text(encoding="utf-8"))
        c2s = cfg.get("centroid_to_super")
        if isinstance(c2s, dict):
            # dict -> list 정규화
            K = len(c2s)
            _centroid_to_super = [int(c2s[str(i)]) for i in range(K)]
        else:
            _centroid_to_super = [int(x) for x in c2s]
        # names(optional)
        names = cfg.get("super_names")
        if names and isinstance(names, dict):
            _super_names = {str(k): v for k, v in names.items()}
        else:
            n_super = max(_centroid_to_super) + 1
            _super_names = {str(i): f"super_{i}" for i in range(n_super)}
        logging.info("[categories] mapping: runtime_categories.json loaded")
        return

    # 폴백: super_map.json + super_names.json
    if not SUPER_MAP_JSON.exists():
        raise FileNotFoundError(f"mapping not found: {RUNTIME_JSON} or {SUPER_MAP_JSON}")

    raw = json.loads(SUPER_MAP_JSON.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        # {"0": 4, "1": 6, ...} → list로 정규화
        K = len(raw)
        _centroid_to_super = [int(raw[str(i)]) for i in range(K)]
    elif isinstance(raw, list):
        _centroid_to_super = [int(x) for x in raw]
    else:
        raise ValueError("super_map.json must be dict or list")

    if SUPER_NAMES_JSON.exists():
        _super_names = json.loads(SUPER_NAMES_JSON.read_text(encoding="utf-8"))
        _super_names = {str(k): v for k, v in _super_names.items()}
    else:
        n_super = max(_centroid_to_super) + 1
        _super_names = {str(i): f"super_{i}" for i in range(n_super)}

    logging.info("[categories] mapping: super_map.json(+names) loaded")


def _init():
    """모델/센트로이드/매핑 리소스 1회 초기화"""
    global _tokenizer, _model, _device, _kmeans

    if _tokenizer is not None:
        return

    model_name = "skt/kobert-base-v1"
    logging.info(f"[categories] loading model: {model_name}")
    # KoBERT는 SentencePiece 기반 → fast=False
    _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    _model = AutoModel.from_pretrained(model_name).eval()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)

    # 센트로이드
    if not CENTROIDS_DEFAULT.exists():
        raise FileNotFoundError(f"centroids not found: {CENTROIDS_DEFAULT}")
    cents = np.load(CENTROIDS_DEFAULT).astype(np.float64)
    _kmeans = KMeans(n_clusters=cents.shape[0], n_init=1, random_state=42)
    _kmeans.cluster_centers_ = cents
    _kmeans._n_threads = 1

    # 매핑
    _load_mapping()

    logging.info(
        f"[categories] init done: K={len(cents)} supers≈{max(_centroid_to_super)+1}"
    )

# ─────────────────────────────────────────
# 핵심: 분류
# ─────────────────────────────────────────
def _embed_word(word: str) -> np.ndarray:
    """
    KoBERT용 안전 임베딩:
    - token_type_ids 제거(모델이 내부 0 세팅) → index 오류 회피
    - attention_mask 기반 mean pooling → subword 분할에도 안정
    """
    with torch.no_grad():
        toks = _tokenizer(word, return_tensors="pt", add_special_tokens=True)
        # token_type_ids가 모델 type_vocab_size와 충돌하는 사례 방지
        toks.pop("token_type_ids", None)
        toks = {k: v.to(_device) for k, v in toks.items()}

        out = _model(**toks).last_hidden_state  # (1, L, H)

        # attention_mask로 가중 평균(패딩 무시)
        if "attention_mask" in toks:
            mask = toks["attention_mask"].unsqueeze(-1).type_as(out)  # (1, L, 1)
        else:
            mask = torch.ones(out.shape[:2], device=out.device).unsqueeze(-1)

        emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (1, H)
        emb = emb[0].detach().cpu().numpy().astype(np.float64)  # (H,)
        return emb


def _predict_super_from_word(word: str):
    emb = _embed_word(word)
    cid = int(_kmeans.predict([emb])[0])
    sid = _centroid_to_super[cid]
    sname = _super_names.get(str(sid), f"super_{sid}")
    return cid, sid, sname

def predict_category(text: str):
    """문장 → 형태소(명사/형용사) → 단어 임베딩 → KMeans → 슈퍼카테고리 다수결"""
    _init()

    morphs = extract_morphs(text)  # [(word,pos), ...]
    picks = []
    for w, pos in morphs:
        if pos not in POS_ALLOWED:
            continue
        try:
            cid, sid, sname = _predict_super_from_word(w)
            picks.append({"word": w, "centroid": cid, "super": sid, "super_name": sname})
        except Exception:
            continue

    # 형태소 단계에서 아무 것도 못 얻었을 때: 공백 토큰 폴백 시도
    if not picks:
        for w in text.split():
            try:
                cid, sid, sname = _predict_super_from_word(w)
                picks.append({"word": w, "centroid": cid, "super": sid, "super_name": sname})
            except Exception:
                continue

    if not picks:
        return "알수없음", []

    counts = Counter(p["super"] for p in picks)
    sid = counts.most_common(1)[0][0]
    sname = _super_names.get(str(sid), f"super_{sid}")
    return sname, picks

# ─────────────────────────────────────────
# Flask 라우트
# ─────────────────────────────────────────
@categories_bp.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json() or {}
    text = data.get("promptContent", "").strip()
    if not text:
        return jsonify({"error": "promptContent is required"}), 400

    label, details = predict_category(text)
    return jsonify({"label": label, "details": details}), 200

# ─────────────────────────────────────────
# 로컬 테스트
# ─────────────────────────────────────────
if __name__ == "__main__":
    _init()
    samples = [
        "야외 농구 코트를 배경으로 덩크슛을 하는 남자",
        "부엌에서 케이크를 장식하는 제빵사",
        "지하철 플랫폼에서 노란 우산을 든 여자",
        "바닷가에서 서핑보드를 타는 소년",
    ]
    for s in samples:
        cat, info = predict_category(s)
        print(f"{s} → {cat} | {info[:3]}")
