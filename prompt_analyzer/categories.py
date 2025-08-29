# AI/prompt_analyzer/categories.py

import os
import json
import logging
from pathlib import Path

import numpy as np
import torch
from flask import Blueprint, request, jsonify
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel

try:
    from .preprocessor import extract_morphs
except ImportError:
    from AI.prompt_analyzer.preprocessor import extract_morphs

# 환경/로그
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.getLogger().setLevel(logging.INFO)

categories_bp = Blueprint("category", __name__, url_prefix="/category")

AI_DIR     = Path(__file__).resolve().parents[1]         
DPDT_DIR   = AI_DIR / "DPDT"
DATA_DIR   = DPDT_DIR / "data"
MODELS_DIR = DPDT_DIR / "models"

CENTROIDS_PATH  = MODELS_DIR / "kmeans_k100" / "centroids.npy"
RUNTIME_JSON    = DATA_DIR / "runtime_categories.json"    
SUPER_MAP_JSON  = DATA_DIR / "super_map.json"            
SUPER_NAMES_JSON= DATA_DIR / "super_names.json"          

# 전역 리소스
_model_name = "klue/bert-base"
_tokenizer = None
_model = None
_device = None
_kmeans = None

_centroid_to_super = None   
_super_names = None           

# 품사 허용( MeCab + Okt )
POS_ALLOWED = {"NNG", "NNP", "NP", "VA", "Noun", "Adjective"}


KW2SUPER_JSON = DATA_DIR / "category_keywords.json"
_kw2super = {}

# 리소스 로드
def _load_super_mapping():
    """runtime_categories.json 우선, 없으면 super_map.json(+super_names.json)"""
    global _centroid_to_super, _super_names

    if RUNTIME_JSON.exists():
        cfg = json.loads(RUNTIME_JSON.read_text(encoding="utf-8"))
        c2s = cfg.get("centroid_to_super")
        if isinstance(c2s, dict):
            K = len(c2s)
            _centroid_to_super = [int(c2s[str(i)]) for i in range(K)]
        else:
            _centroid_to_super = [int(x) for x in c2s]
        names = cfg.get("super_names") or {}
        _super_names = {str(k): v for k, v in names.items()}
        logging.info("[categories] mapping: runtime_categories.json loaded")
        return

    if not SUPER_MAP_JSON.exists():
        raise FileNotFoundError(
            f"super mapping not found: {RUNTIME_JSON} or {SUPER_MAP_JSON}"
        )

    raw = json.loads(SUPER_MAP_JSON.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
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
    global _tokenizer, _model, _device, _kmeans, _kw2super
    if _tokenizer is not None:
        return

    logging.info(f"[categories] loading model: '{_model_name}'")
    _tokenizer = AutoTokenizer.from_pretrained(_model_name, use_fast=True)
    _model = AutoModel.from_pretrained(_model_name).eval()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)

    if not CENTROIDS_PATH.exists():
        raise FileNotFoundError(f"centroids not found: {CENTROIDS_PATH}")
    cents = np.load(CENTROIDS_PATH).astype(np.float64)
    _kmeans = KMeans(n_clusters=cents.shape[0], n_init=1, random_state=42)
    _kmeans.cluster_centers_ = cents
    _kmeans._n_threads = 1

    _load_super_mapping()

    # 키워드 룩업
    if KW2SUPER_JSON.exists():
        cat_kw = json.loads(KW2SUPER_JSON.read_text(encoding="utf-8"))
        _kw2super = {kw: cat for cat, kws in cat_kw.items() for kw in kws}
        logging.info(f"[categories] keyword map loaded ({len(_kw2super)} keys)")
    else:
        _kw2super = {}
        logging.info("[categories] keyword map not found (optional)")

    logging.info(
        f"[categories] init done: K={cents.shape[0]}, supers≈{max(_centroid_to_super)+1}"
    )

# 임베딩 & 예측
def _embed_word(word: str) -> np.ndarray:
    with torch.no_grad():
        toks = _tokenizer(word, return_tensors="pt", add_special_tokens=True)
        toks.pop("token_type_ids", None)
        toks = {k: v.to(_device) for k, v in toks.items()}
        out = _model(**toks).last_hidden_state  # (1, L, H)

        if "attention_mask" in toks:
            mask = toks["attention_mask"].unsqueeze(-1).type_as(out)
        else:
            mask = torch.ones(out.shape[:2], device=out.device).unsqueeze(-1)

        emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (1, H)
        return emb[0].detach().cpu().numpy().astype(np.float64)

def _super_name_from_cid(cid: int) -> str:
    sid = _centroid_to_super[cid]
    return _super_names.get(str(sid), f"super_{sid}")

def _predict_super_from_word(word: str) -> str:
    if word in _kw2super:
        return _kw2super[word]
    emb = _embed_word(word)
    cid = int(_kmeans.predict([emb])[0])
    return _super_name_from_cid(cid)

# API
@categories_bp.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt_id = data.get("promptId")
        text = (data.get("promptContent") or "").strip()

        if not prompt_id:
            return jsonify({"error": "promptId is required"}), 400
        if not text:
            return jsonify({"error": "promptContent is required"}), 400

        _init()

        # 형태소 추출
        morphs = extract_morphs(text)

        results = []
        seen_words = set()

        # 명사/형용사만 카테고리 예측
        for w, pos in morphs:
            if pos not in POS_ALLOWED:
                continue
            if w in seen_words:
                continue
            try:
                sname = _predict_super_from_word(w)
                results.append({
                    "text": w,
                    "classification": sname,
                    "category": sname   
                })
                seen_words.add(w)
            except Exception:
                continue

        # 형태소에서 하나도 못 잡았을 때: 공백 단위 폴백
        if not results:
            for w in text.split():
                w = w.strip()
                if not w or w in seen_words:
                    continue
                try:
                    sname = _predict_super_from_word(w)
                    results.append({
                        "text": w,
                        "classification": sname,
                        "category": sname 
                    })
                    seen_words.add(w)
                except Exception:
                    continue
            
        arr = [
            {
                "text": item["text"],
                "classification": item["classification"],
                "category": item.get("category", item["classification"]),
            }
            for item in results
        ]

        payload = {"results": arr}

        logging.info(
            "[categories] n=%d, preview=%s",
            len(arr), json.dumps(payload, ensure_ascii=False)[:200]
        )
        return jsonify(payload), 200

    except Exception as e:
        logging.exception("category/predict error")
        return jsonify({"error": str(e)}), 500

# 로컬 테스트
if __name__ == "__main__":
    _init()
    samples = [
        "야외 농구 코트를 배경으로 덩크슛을 하는 남자",
        "부엌에서 케이크를 장식하는 제빵사",
        "지하철 플랫폼에서 노란 우산을 든 여자",
        "바닷가에서 서핑보드를 타는 소년",
    ]
    for s in samples:
        outs = []
        for w, pos in extract_morphs(s):
            if pos in POS_ALLOWED:
                try:
                    outs.append((w, _predict_super_from_word(w)))
                except Exception:
                    pass
        print(f"[{s}] → {outs[:6]}")
