# prompt_analyzer/fluency.py

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
    return_detail: bool = False,   # <-- 세부값 리턴 옵션
):

    # 1) 형태소 추출
    token_seqs = [[w for w, _ in extract_morphs(t)] for t in texts]

    # 2) 토크나이저 & 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    bert = AutoModel.from_pretrained(model_name).eval()

    # 3) 임베딩
    id_seqs = []
    for seq in token_seqs[:max_sent]:
        encoded = tokenizer(
            seq,
            is_split_into_words=True,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        id_seqs.append(encoded["input_ids"])

    all_embs = []
    for ids in id_seqs:
        with torch.no_grad():
            out = bert(torch.tensor([ids]))[0]   # (1, L, H)
            emb = out.mean(dim=1).squeeze(0).numpy()  # 평균 풀링
            all_embs.append(emb)
    all_embs = np.stack(all_embs, axis=0).astype(np.float32)

    # 4) KMeans predict
    centroids = np.load(centroids_path).astype(np.float32)
    kmeans = KMeans(n_clusters=centroids.shape[0], n_init=1)
    kmeans.cluster_centers_ = centroids
    kmeans._n_threads = 1
    clusters = kmeans.predict(all_embs)

    # 5) S, K, C 계산
    S = min(len(texts), max_sent) / max_sent
    K = len({tok for seq in token_seqs for tok in seq}) / 1000
    C = len(set(clusters)) / centroids.shape[0]

    # 6) 최종 점수
    score = (
        weight_s * S +
        weight_k * K +
        weight_c * C
    ) / (weight_s + weight_k + weight_c)
    score = float(np.clip(score, 0.0, 1.0))

    if return_detail:
        return score, S, K, C
    return score


# ─────────────────────────────────────
# 로컬 테스트용
# ─────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--centroids", required=True)
    args = p.parse_args()

    texts = [
        "안개 낀 숲길을 홀로 걷는 사람",
        "강아지가 뛰노는 푸른 들판",
        "도시의 밤거리를 달리는 자동차",
        "아이들이 공원에서 뛰어노는 장면",
    ]
    score, S, K, C = compute_fluency(texts, args.centroids, return_detail=True)
    print(f"Fluency Score = {score:.4f}")
    print(f" - S(문장수) = {S:.4f}")
    print(f" - K(고유토큰수) = {K:.4f}")
    print(f" - C(클러스터커버리지) = {C:.4f}")
