# AI/prompt_analyzer/fluency.py

import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from AI.prompt_analyzer.preprocessor import extract_morphs

def compute_fluency(
    texts: list[str],
    centroids_path: str,
    model_name: str = "skt/kobert-base-v1",
    max_sent: int = 1000,
    weight_s: float = 1.0,
    weight_k: float = 1.0,
    weight_c: float = 1.0,
    floor_k: float = 0.20,
):
    # 1) 형태소 → 토큰 시퀀스
    token_seqs = [[w for w, _ in extract_morphs(t)] for t in texts]

    # 2) 임베딩
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name).eval()

    id_seqs = []
    for seq in token_seqs[:max_sent]:
        enc = tokenizer(
            seq,
            is_split_into_words=True,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        id_seqs.append(enc["input_ids"])

    all_embs = []
    for ids in id_seqs:
        with torch.no_grad():
            out = model(torch.tensor([ids]))[0]
            emb = out.mean(dim=1).squeeze(0).numpy()  
            all_embs.append(emb)
    all_embs = np.stack(all_embs, axis=0).astype(np.float32) if all_embs else np.zeros((0, 768), np.float32)

    # 3) KMeans predict
    cents = np.load(centroids_path).astype(np.float32)
    kmeans = KMeans(n_clusters=cents.shape[0], n_init=1)
    kmeans.cluster_centers_ = cents
    kmeans._n_threads = 1
    clusters = kmeans.predict(all_embs) if len(all_embs) else np.array([], dtype=int)

    # 4) S/K/C 계산 (0~1)
    N = len(token_seqs)
    T = sum(len(seq) for seq in token_seqs)
    flat_tokens = [tok for seq in token_seqs for tok in seq]
    U = len(set(flat_tokens))

    # S: 문장수 포화
    tau_s = 8.0
    S = float(1.0 - np.exp(-float(N) / tau_s))

    # K: 어휘 다양성
    if T > 0 and U > 1:
        from collections import Counter
        cnts = np.array(list(Counter(flat_tokens).values()), dtype=np.float64)
        p = cnts / cnts.sum()
        H = -np.sum(p * np.log(p + 1e-12))
        H_norm = H / np.log(min(U, 500) + 1e-12)      
        tau_k = 20.0                                  # 패널티 민감도
        small_pen_k = 1.0 - np.exp(-float(T) / tau_k)
        K = float(np.clip(H_norm * small_pen_k, 0.0, 1.0))
    else:
        K = 0.0

    # C: 클러스터 커버리지
    if len(clusters) > 0:
        from collections import Counter
        K_clusters = int(cents.shape[0])
        counts = np.zeros(K_clusters, dtype=np.float64)
        for k, v in Counter(clusters).items():
            counts[int(k)] = v
        alpha = 0.5
        p_c = (counts + alpha) / (counts.sum() + alpha * K_clusters)
        Hc = -np.sum(p_c * np.log(p_c + 1e-12))
        C_base = Hc / np.log(min(K_clusters, 64) + 1e-12)
        tau_c = 5.0
        small_pen_c = 1.0 - np.exp(-float(N) / tau_c)
        C = float(np.clip(C_base * small_pen_c, 0.0, 1.0))
    else:
        C = 0.0

    # floor_k 스케일 보정 
    if floor_k > 1.0:
        floor_k = floor_k / 100.0

    # 5) 바닥 점수(floor) 적용 (K만)
    if floor_k is not None:
        K = max(K, float(floor_k))

    # 6) 가중합 → 0~100
    wS, wK, wC = float(weight_s), float(weight_k), float(weight_c)
    denom = max(wS + wK + wC, 1e-12)
    score01 = (wS * S + wK * K + wC * C) / denom
    score01 = float(np.clip(score01, 0.0, 1.0))

    return score01 * 100.0, S * 100.0, K * 100.0, C * 100.0

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--centroids", required=True)
    p.add_argument("--floor_k", type=float, default=0.20)
    args = p.parse_args()

    texts = [
        "안개 낀 숲길을 홀로 걷는 사람",
        "강아지가 뛰노는 푸른 들판",
        "도시의 밤거리를 달리는 자동차",
        "아이들이 공원에서 뛰어노는 장면",
    ]
    score, S, K, C = compute_fluency(
        texts, args.centroids, floor_k=args.floor_k
    )
    print(f"Fluency Score = {score:.4f}")
    print(f" - S(문장수) = {S:.4f}")
    print(f" - K(어휘다양성) = {K:.4f}")
    print(f" - C(클러스터커버리지) = {C:.4f}")