# prompt_analyzer/fluency.py

import numpy as np
from sklearn.cluster import KMeans  # 이미 centroids.npy 로 저장해 둔 클러스터 중심 불러올 거예요
from prompt_analyzer.preprocessor import extract_morphs
from transformers import AutoTokenizer
import json

def compute_fluency(
    texts: list[str],                   # <- raw 문자열 리스트
    centroids_path: str,                # KMeans 중심 파일
    model_name: str = "skt/kobert-base-v1",
    max_sent: int = 1000,
    weight_s: float = 1.0,
    weight_k: float = 1.0,
    weight_c: float = 1.0,
) -> float:
    """
    texts: ["안개 낀 숲길...", "다른 문장...", ...]
    0~1 Fluency 점수
    """

    # 1) 형태소 추출
    token_seqs = [ [w for w,_ in extract_morphs(t)] for t in texts ]

    # 2) BERT 토크나이저 로드 & 토큰ID 변환
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
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

    # 3) 임베딩 → 클러스터 할당
    #    (centroids.npy 에 학습해 둔 KMeans cluster centers를 그대로 불러와 predict 만 씁니다)
    centroids = np.load(centroids_path)       # shape = (n_clusters, emb_dim)
    kmeans = KMeans(n_clusters=centroids.shape[0])
    kmeans.cluster_centers_ = centroids
    kmeans._n_threads = 1                     # thread 설정 해제
    # (fast predict)
    # token_ids 하나하나를 임베딩하려면 embedding matrix가 필요하지만, 
    # 여기서는 미리 토크나이저/모델의 임베딩 레이어를 로드하셔야 합니다.
    # 예시로 HuggingFace의 KorBERT embedding layer를 사용:
    from transformers import AutoModel
    bert = AutoModel.from_pretrained(model_name)
    bert.eval()

    all_embs = []
    for ids in id_seqs:
        import torch
        with torch.no_grad():
            out = bert(torch.tensor([ids]))[0]   # (1, seq_len, hidden_dim)
            # [CLS] 토큰 임베딩만 쓸 수도 있고, 전체 평균 풀링도 가능합니다.
            cls_emb = out[0,0].numpy()           # (hidden_dim,)
            all_embs.append(cls_emb)
    all_embs = np.stack(all_embs, axis=0)      # (n_sent, hidden_dim)

    clusters = kmeans.predict(all_embs)        # (n_sent,)

    # 4) S, K, C 계산
    S = min(len(texts), max_sent) / max_sent   # 문장 수 정규화
    K = len({tok for seq in token_seqs for tok in seq}) / 1000  # 예시로 1000개 최대
    C = len(set(clusters)) / centroids.shape[0]

    # 5) 가중합 & 0~1 클리핑
    score = (
        weight_s * S +
        weight_k * K +
        weight_c * C
    ) / (weight_s + weight_k + weight_c)
    return float(np.clip(score, 0.0, 1.0))
