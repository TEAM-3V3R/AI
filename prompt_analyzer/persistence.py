# AI/prompt_analyzer/persistence.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import Counter

from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel

# 형태소 기반 토큰(어휘 집중도 R 계산용)
try:
    from AI.prompt_analyzer.preprocessor import extract_morphs
except Exception:
    from prompt_analyzer.preprocessor import extract_morphs  # type: ignore


# -----------------------------
# 임베딩 헬퍼: 강력 방어 + 패딩 무시 평균 풀링
# -----------------------------
@torch.no_grad()
def _embed_sentences(
    texts: List[str],
    tokenizer,
    model,
    device,
    max_length: int = 128,
) -> np.ndarray:
    """
    문장 -> 임베딩 (N,H)
    - token_type_ids: 항상 0으로 강제 (KoBERT 안정화)
    - input_ids: 항상 vocab 범위로 clamp (out-of-range 방지)
    - 실패 시 token_type_ids 제거하고 재시도
    - 패딩 제외 평균 풀링
    """
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        # BertModel 계열이면 보통 존재하지만, 안전하게 처리
        hidden_size = model.embeddings.word_embeddings.embedding_dim

    if len(texts) == 0:
        return np.zeros((0, hidden_size), dtype=np.float32)

    # 1) 인코딩 (여기선 use_fast 등 kwargs 금지)
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_attention_mask=True,
    )

    # 2) 항상 segment를 0으로 강제 (범위/타입 안전)
    #    → enc에 있든 없든 덮어쓴다.
    ttids = torch.zeros_like(enc["input_ids"], dtype=torch.long)
    enc["token_type_ids"] = ttids

    # 3) 디바이스 이동
    enc = {k: v.to(device) for k, v in enc.items()}

    # 4) vocab 범위로 clamp (word_embeddings out-of-range 방지)
    V = model.embeddings.word_embeddings.num_embeddings
    enc["input_ids"] = enc["input_ids"].clamp_(0, V - 1)

    # 5) Forward with robust fallback
    def _forward_and_pool(e):
        out = model(**e).last_hidden_state  # (B, L, H)
        mask = e["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        masked = out * mask
        sum_vec = masked.sum(dim=1)              # (B, H)
        len_vec = mask.sum(dim=1).clamp_min(1.0) # (B, 1)
        pooled = sum_vec / len_vec               # (B, H)
        return pooled

    try:
        sent_embs = _forward_and_pool(enc)
    except IndexError:
        # 드물게 token_type_embeddings에서 범위 오류가 나면 세그먼트 자체 제거 후 재시도
        enc_no_seg = {k: v for k, v in enc.items() if k != "token_type_ids"}
        sent_embs = _forward_and_pool(enc_no_seg)

    return sent_embs.detach().cpu().numpy().astype(np.float32)


# ------------------------------------
# 핵심: Persistence (Simpson/Herfindahl)
# ------------------------------------
def compute_persistence(
    texts: List[str],
    centroids_path: str,
    model_name: str = "skt/kobert-base-v1",
    *,
    # 가중치(0~1 스케일에서 가중합)
    weight_s: float = 1.0,
    weight_r: float = 1.0,
    weight_f: float = 1.0,
    # 포화/소표본 패널티 민감도
    tau_s: float = 3.0,    # S: 문장 수 포화
    tau_r: float = 1.0,   # R: 토큰 수 소표본 패널티
    tau_f: float = 5.0,   # F: 문장 수 소표본 패널티
    # 선택적 리소스 재사용(속도↑): {'tokenizer','model','device','centroids'}
    resources: Optional[Dict] = None,
) -> Tuple[float, float, float, float]:
    """
    반환: (score, S, R, F)  모두 0~100 스케일
      - S: size (문장 수 포화)
      - R: lexical concentration — Simpson index(∑p_i²) × 소표본 패널티
      - F: cluster focus        — Simpson index(∑q_k²) × 소표본 패널티
    """
    # ── 모델/토크나이저/디바이스/센트로이드 ──────────────────────
    if resources and {"tokenizer", "model", "device"} <= set(resources.keys()):
        tokenizer = resources["tokenizer"]
        model = resources["model"]
        device = resources["device"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModel.from_pretrained(model_name).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    if resources and isinstance(resources.get("centroids"), np.ndarray):
        cents = resources["centroids"].astype(np.float32, copy=False)
    else:
        cents = np.load(centroids_path)
        if cents.dtype != np.float32:
            cents = cents.astype(np.float32, copy=False)

    K_clusters = int(cents.shape[0])

    # ── S: 문장 수 포화 (0~1) ───────────────────────────────────
    N = len(texts)
    modifiers = 0
    for t in texts:
        morphs = extract_morphs(t)
        for w, pos in morphs:
            if pos in ("VA", "MAG", "MM",  "Adjective", "Adverb", "Determiner"):  # 형용사, 부사, 관형사
                modifiers += 1
    
    S = 1.0 - np.exp(-float(modifiers) / tau_s)

    # ── R: 어휘 집중도(심프슨) ─────────────────────────────────
    flat_tokens: List[str] = []
    for t in texts:
        toks = [w for (w, _pos) in extract_morphs(t)]
        flat_tokens.extend(toks)

    T = len(flat_tokens)
    if T > 0:
        cnts = np.array(list(Counter(flat_tokens).values()), dtype=np.float64)
        beta = 1.3  # 1.0(기존) <= beta; 1.1~1.5 권장
        w = cnts ** beta
        p = w / w.sum()
        simpson_R = np.sum(p * p)
        small_pen_r = 1.0 - np.exp(-float(T) / float(tau_r))
        R = (simpson_R * small_pen_r)

        # 최소 0.2 보장
        R = max(R, 0.2)

        entropy = -np.sum(p * np.log(p+1e-12)) / np.log(len(p)+1e-12)
        R = 0.5 * simpson_R + 0.5 * entropy
    else:
        R = 0.0

    # ── F: 클러스터 집중도(심프슨) ─────────────────────────────
    embs = _embed_sentences(texts, tokenizer, model, device, max_length=128)  # (N,H)
    if embs.shape[0] > 0:
        km = KMeans(n_init=1, random_state=42, n_clusters=K_clusters)
        km.cluster_centers_ = cents
        km._n_threads = 1
        labels = km.predict(embs)  # (N,)

        counts = np.zeros(K_clusters, dtype=np.float64)
        for k, v in Counter(labels).items():
            idx = int(k)
            if 0 <= idx < K_clusters:
                counts[idx] = v
        q = counts / max(counts.sum(), 1e-12)
        simpson_F = float(np.sum(q * q))
        small_pen_f = 1.0 - np.exp(-float(N) / float(tau_f))
        F = float(np.clip(simpson_F * small_pen_f, 0.0, 1.0))
    else:
        F = 0.0

    # ── 최종 점수 (가중합 → 0~100) ─────────────────────────────
    wS, wR, wF = float(weight_s), float(weight_r), float(weight_f)
    denom = max(wS + wR + wF, 1e-12)
    score01 = (wS * S + wR * R + wF * F) / denom
    score01 = float(np.clip(score01, 0.0, 1.0))

    return score01 * 100.0, S * 100.0, R * 100.0, F * 100.0


# -----------------------------
# 로컬 테스트 (python -m ... )
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Persistence (Simpson/Herfindahl)")
    ap.add_argument("--centroids", required=True, help="centroids.npy 경로")
    ap.add_argument("--model", default="skt/kobert-base-v1")
    ap.add_argument("--w_s", type=float, default=1.0)
    ap.add_argument("--w_r", type=float, default=1.0)
    ap.add_argument("--w_f", type=float, default=1.0)
    ap.add_argument("--tau_s", type=float, default=8.0)
    ap.add_argument("--tau_r", type=float, default=30.0)
    ap.add_argument("--tau_f", type=float, default=20.0)
    args = ap.parse_args()

    # 샘플
    texts = [
        "안개 낀 숲길을 홀로 걷는 사람",
        "강아지가 뛰노는 푸른 들판",
        "도시의 밤거리를 달리는 자동차",
        "아이들이 공원에서 뛰어노는 장면",
    ]

    score, S, R, F = compute_persistence(
        texts,
        centroids_path=args.centroids,
        model_name=args.model,
        weight_s=args.w_s,
        weight_r=args.w_r,
        weight_f=args.w_f,
        tau_s=args.tau_s,
        tau_r=args.tau_r,
        tau_f=args.tau_f,
    )
    print(f"Persistence Score = {score:.4f}")
    print(f" - S(size)                 = {S:.4f}")
    print(f" - R(lexical concentration)= {R:.4f}")
    print(f" - F(cluster focus)        = {F:.4f}")
