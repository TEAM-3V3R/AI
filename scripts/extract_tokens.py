# scripts/extract_word_embeddings.py

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

def main():
    p = argparse.ArgumentParser(
        description="단어 리스트 → BERT 단어 임베딩 추출 (tokens_word.jsonl)"
    )
    p.add_argument("--input", "-i", required=True,
                   help="입력 파일: 단어 목록 (예: words.txt)")
    p.add_argument("--output", "-o", required=True,
                   help="출력 파일: JSONL 형식 (예: tokens_word.jsonl)")
    p.add_argument("--model", "-m", default="skt/kobert-base-v1",
                   help="사용할 HuggingFace 모델 이름")
    p.add_argument("--use_fast", action="store_false",
                   help="Fast tokenizer 비활성화")

    args = p.parse_args()

    inp_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    # 1) 모델 & 토크나이저 로드
    print("⏳ Loading tokenizer and model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast
    )
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with inp_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        seen = set()  # 중복 단어 제거용
        for line in fin:
            word = line.strip()
            if not word or word in seen:
                continue
            seen.add(word)

            encoded = tokenizer(
                word,
                add_special_tokens=True,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                emb = outputs.pooler_output.squeeze(0)  # → (hidden_size,)

            record = {
                "token": word,
                "emb": emb.cpu().tolist()
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ 단어 임베딩 추출 완료 → {out_path}")

if __name__ == "__main__":
    main()
