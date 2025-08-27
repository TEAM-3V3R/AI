# AI/prompt_analyzer/persistence.py
import argparse, os, logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.getLogger().setLevel(logging.INFO)

def compute_persistence(texts, centroids_path, model_name="skt/kobert-base-v1", max_len=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # KoBERT = SentencePiece 기반
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    centroids = np.load(centroids_path).astype(np.float32)
    all_clusters = []

    with torch.no_grad():
        for sent in texts:
            enc = tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding="max_length"
            )
            if "token_type_ids" in enc:
                enc.pop("token_type_ids")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).last_hidden_state  # (1, L, H)

            pad_id = tokenizer.pad_token_id or 0
            mask = (enc["input_ids"][0] != pad_id).cpu().numpy()
            valid_embs = out[0][mask].cpu().numpy().astype(np.float32)

            norms_c = np.linalg.norm(centroids, axis=1, keepdims=True)
            for e in valid_embs:
                dot = centroids @ e
                norm = norms_c.squeeze() * np.linalg.norm(e)
                cos = dot / (norm+1e-12)
                cid = int(np.argmax(cos))
                all_clusters.append(cid)

    if not all_clusters:
        return 0.0, 0.0, 0.0, 0.0

    uniq, counts = np.unique(all_clusters, return_counts=True)
    freqs = counts / counts.sum()
    entropy = -np.sum(freqs * np.log(freqs+1e-12))
    max_entropy = np.log(len(freqs)) if len(freqs) > 1 else 1.0
    persistence = entropy / (max_entropy+1e-12)

    S = len(texts)/1000
    K = len(uniq)/centroids.shape[0]
    C = persistence
    score = (S+K+C)/3
    return float(score), float(S), float(K), float(C)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--centroids", required=True)
    args = p.parse_args()

    texts = [
        "안개 낀 숲길을 홀로 걷는 사람",
        "강아지가 뛰노는 푸른 들판",
        "도시의 밤거리를 달리는 자동차",
        "아이들이 공원에서 뛰어노는 장면",
    ]
    score,S,K,C = compute_persistence(texts, args.centroids)
    print(f"Persistence Score = {score:.4f}")
    print(f" - S = {S:.4f}")
    print(f" - K = {K:.4f}")
    print(f" - C = {C:.4f}")
