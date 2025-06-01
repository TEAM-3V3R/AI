# prompt_analyzer/persistence.py

import os
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

def compute_persistence(
    texts_or_idlists,        # “문장 리스트(str)” or “ID 리스트(list of list of ints)”
    centroids_path: str,
    model_name: str = "skt/kobert-base-v1",
):
    """
    Persistence 점수 계산 (토큰 수준 클러스터링 버전)
    - texts_or_idlists: 
        • 문자열 리스트라면, 내부에서 tokenizer를 거쳐 ID로 변환
        • 이미 ID 배열 리스트([ [517,0,490,…], [517,491,…], … ])라면 그대로 사용
    """

    # 1) BERT 모델 + 토크나이저 불러오기
    device = torch.device("cpu")
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # AutoTokenizer도 미리 불러 두면 편합니다.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2) 센트로이드 로딩
    centroids = np.load(centroids_path)  # shape = (n_clusters, hidden_dim)

    # 3) 입력 타입 검사(문장(str) vs ID 리스트)
    is_idlists = isinstance(texts_or_idlists[0], (list, tuple))

    assigned_clusters = []

    with torch.no_grad():
        for entry in texts_or_idlists:
            if is_idlists:
                # entry가 “이미 ID 리스트”인 경우
                token_ids = torch.tensor([entry], dtype=torch.long, device=device)
                attention_mask = (token_ids != 0).long()
            else:
                # entry가 “문자열 문장”인 경우 → 미리 형태소 레벨 clean_text나 
                # 띄어쓰기 교정이 되어 있어서, 단순히 split()해서 사용한다고 가정
                encoded = tokenizer(
                    entry.split(), 
                    is_split_into_words=True,
                    add_special_tokens=True,
                    padding=True,
                    # truncation=True,
                    return_tensors="pt"
                )
                token_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

            # (A) [batch_size=1, seq_len] → BERT 임베딩
            outputs = model(input_ids=token_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

            # (B) 각 토큰(t=0~seq_len-1)별로 벡터 추출
            #     last_hidden[0, t, :] 형태로 hidden_dim 벡터 얻음
            #     그러면 총 seq_len개 토큰별 임베딩이 생김
            seq_len = last_hidden.shape[1]
            for t in range(seq_len):
                token_emb = last_hidden[0, t, :].cpu().numpy()  # (hidden_dim,)
                
                # (C) 각 토큰 임베딩과 모든 centroids 사이의 코사인 유사도 계산
                #     centroids: (n_clusters, hidden_dim)
                #     token_emb: (hidden_dim,)
                #     → 코사인(sim) = (centroids · token_emb) / (‖centroids‖ · ‖token_emb‖)
                dot = centroids @ token_emb   # (n_clusters,)
                norms = np.linalg.norm(centroids, axis=1) * np.linalg.norm(token_emb)
                cosines = dot / (norms + 1e-12)

                # (D) 가장 가까운 클러스터 인덱스 취함
                cluster_id = int(np.argmax(cosines))
                assigned_clusters.append(cluster_id)

    # 4) “토큰 단위로 할당된 cluster_id들”을 기반으로 분포 계산
    assigned_clusters = np.array(assigned_clusters)
    if assigned_clusters.size == 0:
        return 0.0

    unique, counts = np.unique(assigned_clusters, return_counts=True)
    freqs = counts / counts.sum()

    # 5) 엔트로피 → 0~1 정규화 (cluster 개수 대신 “토큰 분포에 나온 클러스터 수” 사용)
    entropy = -np.sum(freqs * np.log(freqs + 1e-12))
    max_entropy = np.log(len(freqs)) if len(freqs) > 1 else 1.0
    persistence_score = entropy / (max_entropy + 1e-12)

    return float(persistence_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Persistence(지속성) 점수를 계산합니다."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Persistence를 계산할 입력. ■텍스트(.txt) or ■ID 배열(.txt) 파일"
    )
    parser.add_argument(
        "-c", "--centroids",
        required=True,
        help="학습된 클러스터 센트로이드 파일 경로 (예: data/centroids.npy)"
    )
    parser.add_argument(
        "-m", "--model",
        default="skt/kobert-base-v1",
        help="HuggingFace BERT 모델 이름"
    )
    args = parser.parse_args()

    from pathlib import Path
    input_path = Path(args.input)
    lines = [l.strip() for l in input_path.open(encoding="utf8") if l.strip()]

    first = None
    try:
        first = json.loads(lines[0])
    except:
        first = lines[0]

    if isinstance(first, list):
        # “tokens.txt”처럼 ID 배열이 저장된 경우
        id_lists = [json.loads(line) for line in lines]
        score = compute_persistence(
            id_lists,
            centroids_path=args.centroids,
            model_name=args.model
        )
    else:
        # 순수 텍스트(“한 줄에 한 문장”)인 경우
        score = compute_persistence(
            texts_or_idlists=lines,
            centroids_path=args.centroids,
            model_name=args.model
        )

    print(f"Persistence 점수: {score:.4f}")
