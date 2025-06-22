# prompt_analyzer/categories.py

from flask import Blueprint, request, jsonify
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from prompt_analyzer.preprocessor import extract_morphs

categories_bp = Blueprint("category", __name__, url_prefix="/category")

# 모델·리소스 초기화 (앱이 시작될 때 한 번만)
_model_name = "klue/bert-base"
_tokenizer, _model, _device = None, None, None
_kmeans, _centroid_to_cat = None, None
_keyword_to_cat = None

def _init():
    global _tokenizer, _model, _device, _kmeans, _centroid_to_cat, _keyword_to_cat
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_model_name, use_fast=True)
        _model     = AutoModel.from_pretrained(_model_name).eval()
        _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)

        centroids = np.load(Path("DPDT/data/word_centroids.npy")).astype(np.float64)
        _kmeans    = KMeans(n_clusters=centroids.shape[0])
        _kmeans.cluster_centers_ = centroids
        _kmeans._n_threads       = 1

        _centroid_to_cat = json.loads(
            Path("DPDT/data/word_centroid_categories.json")
                .read_text(encoding="utf-8")
        )

        cat_kw = json.loads(
            Path("DPDT/data/category_keywords.json")
                .read_text(encoding="utf-8")
        )
        _keyword_to_cat = {kw:cat for cat,kws in cat_kw.items() for kw in kws}

@categories_bp.route("/predict", methods=["POST"])
def predict_route():
    data     = request.get_json() or {}
    promptId = data.get("promptId")
    text     = data.get("promptContent", "").strip()

    if not promptId:
        return jsonify({"error": "promptId is required"}), 400
    if not text:
        return jsonify({"error": "promptContent is required"}), 400

    _init()

    # extract_morphs가 (단어, 품사) 리스트를 리턴
    morphs = extract_morphs(text)

    results = []
    seen_cats = set()  # 이미 추가된 카테고리 저장
    with torch.no_grad():
        for word, pos in morphs:
            # **명사(NNG, NNP, NP) 또는 형용사(VA)만 처리**
            if pos not in ("NNG","NNP","NP","VA"):
                continue

            # --- (1) 사전 기반 룩업 우선 ---
            if word in _keyword_to_cat:
                results.append({
                    "text": word,
                    "classification": _keyword_to_cat[word]
                })
                cat = _keyword_to_cat[word]
                if cat not in seen_cats:
                    results.append({"text": word, "classification": cat})
                    seen_cats.add(cat)
                continue

            # --- (2) 임베딩→KMeans 매칭 ---
            toks = _tokenizer(word, return_tensors="pt", add_special_tokens=True)
            toks = {k: v.to(_device) for k, v in toks.items()}
            out  = _model(**toks).last_hidden_state
            emb  = out[0, 0].cpu().numpy().astype(np.float64)
            cid  = int(_kmeans.predict([emb])[0])
            cat  = _centroid_to_cat.get(str(cid), "알수없음")

            results.append({
                "text": word,
                "classification": cat
            })
            if cat not in seen_cats:
                results.append({"text": word, "classification": cat})
                seen_cats.add(cat)

    return jsonify({
        "promptId":   promptId,
        "results":    results
    }), 200
