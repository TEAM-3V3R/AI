# 센트로이드 학습 스크립트
# scripts/train_centroids.py

import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import argparse

def main():
    p = argparse.ArgumentParser(
        description="토큰 임베딩으로 KMeans 클러스터링 → 센트로이드 저장"
    )
    p.add_argument("--input", "-i",
                   required=True,
                   help="DPDT/data/tokens.txt (ids+emb JSON per line)")
    p.add_argument("--output", "-o",
                   required=True,
                   help="출력할 센트로이드 파일 (예: DPDT/data/centroids.npy)")
    p.add_argument("--n-clusters", "-k",
                   type=int,
                   default=50,
                   help="클러스터 개수 (default=50)")
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 임베딩 로드
    embs = []
    with input_path.open("r", encoding="utf8") as fin:
        for line in fin:
            data = json.loads(line)
            emb = data.get("emb", None)
            if emb:
                embs.append(emb)
    embs = np.array(embs, dtype=np.float32)
    print(f"🔸 Loaded {len(embs)} embeddings of dim {embs.shape[1]}")

    # 2) KMeans 학습
    print(f"🔸 Training KMeans with k={args.n_clusters}...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    kmeans.fit(embs)
    centroids = kmeans.cluster_centers_
    print("✅ KMeans training complete")

    # 3) 센트로이드 저장
    np.save(output_path, centroids)
    print(f"✅ Saved centroids to {output_path}")

if __name__ == "__main__":
    main()
