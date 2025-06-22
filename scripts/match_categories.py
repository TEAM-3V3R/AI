# scripts/match_categories.py

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path


def main():
    # 1) 외부 JSON에서 카테고리별 대표 키워드 로드
    cat_json = Path("DPDT/data/category_keywords.json")
    if not cat_json.exists():
        raise FileNotFoundError(f"Category JSON not found: {cat_json}")
    category_keywords = json.loads(cat_json.read_text(encoding="utf-8"))

    # 2) 모델 로드 (WordPiece 기반 KLUE BERT 권장)
    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3) 각 카테고리 키워드 평균 임베딩 계산
    category_vecs = {}
    for cat, words in category_keywords.items():
        toks = tokenizer(words, padding=True, truncation=True, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            out = model(**toks).last_hidden_state  # (N, seq_len, hidden)
        mask = toks["attention_mask"].unsqueeze(-1)
        emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        category_vecs[cat] = emb.mean(dim=0).cpu().numpy()

    # 4) centroids 불러오기
    centroids_path = Path("DPDT/data/word_centroids.npy")
    if not centroids_path.exists():
        raise FileNotFoundError(f"Centroids file not found: {centroids_path}")
    centroids = np.load(centroids_path)

    # 5) 매핑: 각 centroid -> 가장 유사한 카테고리
    centroid_to_cat = {}
    for idx, c in enumerate(centroids):
        sims = {cat: cosine_similarity([c], [vec])[0, 0]
                for cat, vec in category_vecs.items()}
        best = max(sims, key=sims.get)
        centroid_to_cat[str(idx)] = best

    # 6) 결과 저장
    out_path = Path("DPDT/data/word_centroid_categories.json")
    out_path.write_text(json.dumps(centroid_to_cat, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ 카테고리 매핑 완료 →", out_path)


if __name__ == "__main__":
    main()
