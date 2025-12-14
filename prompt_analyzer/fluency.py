# 유연성 점수 계산

import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import List, Dict, Any
from prompt_analyzer.preprocessor import extract_morphs

# 함수 시그니처 / 입력값
def compute_fluency(
    texts: list[str],
    centroids_path: str,
    model_name: str = "skt/kobert-base-v1",
    max_sent: int = 1000,
    weight_s: float = 1.0,
    weight_k: float = 1.0,
    weight_c: float = 1.0,
    floor_k: float = 0.20,
    resources: Dict[str, Any] | None = None,

    # 반복 패널티 옵션
    rep_mode: str = "sentence",      
    rep_strength: float = 1.0,
    rep_floor: float = 0.15,
):
    # resources 준비 (API 캐시 구조와 호환)
    if resources is None:
        raise RuntimeError("resources must be provided")

    tokenizer = resources["tokenizer"]
    model = resources["model"]
    device = resources["device"]
    cents = resources["centroids"]   # np.ndarray (K, 768)

    model.eval()

    # 1) 형태소 → 토큰 시퀀스
    token_seqs = [[w for w, _ in extract_morphs(t)] for t in texts][:max_sent]

    # 2) BERT 임베딩
    embs = []
    for seq in token_seqs:
        if not seq:
            continue

        enc = tokenizer(
            seq,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc).last_hidden_state
            emb = out.mean(dim=1).squeeze(0)
            embs.append(emb.detach().cpu().numpy())

    if embs:
        embs = np.stack(embs).astype(np.float32)
    else:
        embs = np.zeros((0, cents.shape[1]), dtype=np.float32)

    # 3) KMeans predict - 아이디어 클러스터 예측
    if len(embs) > 0:
        kmeans = KMeans(n_clusters=cents.shape[0], n_init=1)
        kmeans.cluster_centers_ = cents
        kmeans._n_threads = 1
        clusters = kmeans.predict(embs)
    else:
        clusters = np.array([], dtype=int)

    # 4) S/K/C 계산 (0~1)
    N = len(token_seqs)
    flat_tokens = [tok for seq in token_seqs for tok in seq]
    T = len(flat_tokens)
    U = len(set(flat_tokens))

    # S: 문장 수 기반 다양성
    tau_s = 8.0
    S = 1.0 - np.exp(-float(N) / tau_s) if N > 0 else 0.0

    # K: 어휘 다양성 + 짧은 입력 패널티
    if T > 1 and U > 1:
        from collections import Counter
        cnts = np.array(list(Counter(flat_tokens).values()), dtype=np.float64)
        p = cnts / cnts.sum()
        H = -np.sum(p * np.log(p + 1e-12))
        H_norm = H / np.log(min(U, 500) + 1e-12)

        tau_k = 20.0
        small_pen_k = 1.0 - np.exp(-float(T) / tau_k)
        K = float(np.clip(H_norm * small_pen_k, 0.0, 1.0))
    else:
        K = 0.0

    # C: 클러스터 커버리지 + 짧은 입력 패널티
    if len(clusters) > 0:
        from collections import Counter
        Kc = cents.shape[0]
        counts = np.zeros(Kc, dtype=np.float64)
        for k, v in Counter(clusters).items():
            counts[int(k)] = v

        alpha = 0.5
        p_c = (counts + alpha) / (counts.sum() + alpha * Kc)
        Hc = -np.sum(p_c * np.log(p_c + 1e-12))
        C_base = Hc / np.log(min(Kc, 64) + 1e-12)

        tau_c = 5.0
        small_pen_c = 1.0 - np.exp(-float(N) / tau_c)
        C = float(np.clip(C_base * small_pen_c, 0.0, 1.0))
    else:
        C = 0.0

    # 5) 바닥 점수(floor) 적용 (K만)
    if floor_k > 1.0:
        floor_k = floor_k / 100.0
    if floor_k is not None:
        K = max(K, float(floor_k))

    # 6) 가중합 → 0~100점 변환
    wS, wK, wC = float(weight_s), float(weight_k), float(weight_c)
    denom = max(wS + wK + wC, 1e-12)
    score01 = (wS * S + wK * K + wC * C) / denom
    score01 = float(np.clip(score01, 0.0, 1.0))

    # 7) 반복 패널티 적용: score01 *= R
    if rep_mode == "sentence" and rep_strength > 0.0:
        sent_keys = [" ".join(seq) for seq in token_seqs if seq]
        uniq_sent = len(set(sent_keys))
        sent_r = uniq_sent / len(sent_keys) if sent_keys else 1.0

        R = np.clip(sent_r, rep_floor, 1.0)
        R = R ** rep_strength

        score01 *= R
        score01 = float(np.clip(score01, 0.0, 1.0))

    return score01 * 100.0, S * 100.0, K * 100.0, C * 100.0