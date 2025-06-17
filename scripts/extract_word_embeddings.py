# scripts/extract_word_embeddings.py

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

# 경로 설정
input_path = Path("DPDT/data/corpus_filtered.txt")
output_path = Path("DPDT/data/tokens_word.jsonl")

# 모델 로드
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 고유 단어 추출
unique_words = set()
with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        words = line.strip().split()
        unique_words.update(words)

# 임베딩 추출 및 저장
with output_path.open("w", encoding="utf-8") as fout:
    for word in sorted(unique_words):
        if not word.strip():
            continue
        tokens = tokenizer(word, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            output = model(**tokens)
            emb = output.last_hidden_state[:, 0, :]  # CLS 임베딩
        fout.write(json.dumps({
            "token": word,
            "emb": emb.squeeze(0).cpu().tolist()
        }, ensure_ascii=False) + "\n")

print(f"✅ tokens_word.jsonl 저장 완료 ({len(unique_words)}개 단어)")
