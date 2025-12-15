from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from collections import Counter

from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from prompt_analyzer.preprocessor import extract_morphs


@torch.no_grad()

# 문장 임베딩 생성기 - F 계산용
def _embed_sentences(
    texts: List[str],
    tokenizer,
    model,
    device,
    max_length: int = 128,
) -> np.ndarray:
    """문장 → 임베딩 (패딩 제외 평균 풀링)"""
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = model.embeddings.word_embeddings.embedding_dim

    if not texts:
        return np.zeros((0, hidden_size), dtype=np.float32)

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        add_special_tokens=True,
    )

    enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
    enc = {k: v.to(device) for k, v in enc.items()}

    V = model.embeddings.word_embeddings.num_embeddings
    enc["input_ids"].clamp_(0, V - 1)

    out = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    return pooled.cpu().numpy().astype(np.float32)

# persistence 계산
def compute_persistence(
    texts: List[str],
    centroids_path: str,
    model_name: str = "skt/kobert-base-v1",
    *,
    weight_s: float = 1.0,
    weight_r: float = 1.0,
    weight_f: float = 1.0,
    tau_s: float = 3.0,
    tau_r: float = 20.0,
    tau_f: float = 5.0,
    r_floor: float = 0.2,
    resources: Optional[Dict] = None,
) -> Tuple[float, float, float, float]:

    # resources (API 캐시 호환)
    if resources and {"tokenizer", "model", "device"} <= resources.keys():
        tokenizer = resources["tokenizer"]
        model = resources["model"]
        device = resources["device"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModel.from_pretrained(model_name).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    if resources and isinstance(resources.get("centroids"), np.ndarray):
        cents = resources["centroids"]
    else:
        cents = np.load(centroids_path).astype(np.float32)

    K_clusters = cents.shape[0]

    # S: 수식어 기반 심화
    modifier_count = 0
    for t in texts:
        for _, pos in extract_morphs(t):
            if pos in ("VA", "MAG", "MM", "Adjective", "Adverb", "Determiner"):
                modifier_count += 1

    S = 1.0 - np.exp(-modifier_count / tau_s) if modifier_count > 0 else 0.0

    # R: 어휘 집중도 (Simpson)
    tokens: List[str] = []
    for t in texts:
        tokens.extend(w for w, _ in extract_morphs(t))

    T = len(tokens)
    if T > 1:
        cnts = np.array(list(Counter(tokens).values()), dtype=np.float64)
        p = cnts / cnts.sum()

        # 반복 집중도 확인
        simpson = float(np.sum(p * p))
        min_simpson = 1.0 / T
        R0 = (simpson - min_simpson) / (1.0 - min_simpson) 
        R0 = float(np.clip(R0, 0.0, 1.0))

        # 길이 보정
        small_pen = 1.0 - np.exp(-T / tau_r)

        # 낮으면 floor 고정, 높으면 상승
        gate = 0.10
        if R <= gate : 
            R = r_floor
        else : 
            up = ((R0 - gate) / (1.0 - gate)) * small_pen_
            R = r_floor + (1.0 - r_floor) * float(np.clip(up, 0.0, 1.0))
    else:
        R = 0.0

    # F: 의미 클러스터 집중도
    embs = _embed_sentences(texts, tokenizer, model, device)
    if len(embs) > 0:
        km = KMeans(n_clusters=K_clusters, n_init=1)
        km.cluster_centers_ = cents
        km._n_threads = 1
        labels = km.predict(embs)

        counts = np.zeros(K_clusters, dtype=np.float64)
        for k, v in Counter(labels).items():
            counts[int(k)] = v

        q = counts / counts.sum()
        simpson_f = float(np.sum(q * q))
        small_pen_f = 1.0 - np.exp(-len(texts) / tau_f)
        F = np.clip(simpson_f * small_pen_f, 0.0, 1.0)
    else:
        F = 0.0

    # 최종 점수 
    denom = max(weight_s + weight_r + weight_f, 1e-12)
    score01 = (weight_s * S + weight_r * R + weight_f * F) / denom

    return (
        score01 * 100.0,
        S * 100.0,
        R * 100.0,
        F * 100.0,
    )
