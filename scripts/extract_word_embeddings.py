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
            outputs   = model(**tokens)
            all_hidden = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
            # [CLS]=0, [SEP]=-1 제외하고 실제 서브워드 임베딩 평균
            token_embs = all_hidden[1:-1]  
            if token_embs.shape[0] == 0:
                # 만약 단어가 하나의 토큰으로만 분절됐다면, 그 토큰 하나를 사용
                word_emb = all_hidden[1]
            else:
                word_emb = token_embs.mean(0)
            emb = word_emb.unsqueeze(0)  # (1, hidden_dim)
        fout.write(json.dumps({
            "token": word,
            "emb": emb.squeeze(0).cpu().tolist()
        }, ensure_ascii=False) + "\n")

print(f"✅ tokens_word.jsonl 저장 완료 ({len(unique_words)}개 단어)")
