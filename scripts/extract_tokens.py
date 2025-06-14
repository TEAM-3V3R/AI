# scripts/extract_tokens.py

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

def main():
    p = argparse.ArgumentParser(
        description="corpus_filtered.txt → BERT token IDs & sentence embeddings 파일(tokens.jsonl) 생성"
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="DPDT/data/corpus_filtered.txt (전처리된 형태소 목록, 공백 분리)"
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        help="DPDT/data/tokens_and_embs.jsonl"
    )
    p.add_argument(
        "--model", "-m",
        default="skt/kobert-base-v1",
        help="HuggingFace 모델 이름 (예: skt/kobert-base-v1)"
    )
    p.add_argument(
        "--use_fast",
        action="store_false",
        help="Fast tokenizer 변환을 비활성화하려면 이 플래그를 추가하세요"
    )
    args = p.parse_args()

    inp_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    # 1) 토크나이저 & 모델 로드
    print("⏳ Loading tokenizer and model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast
    )
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    # (GPU 사용 가능하면 GPU로 옮기기)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with inp_path.open("r", encoding="utf8") as fin, \
         out_path.open("w", encoding="utf8") as fout:
        for line in fin:
            sent = line.strip()
            if not sent:
                continue

            # 2) 토크나이즈 → token IDs + attention mask
            encoded = tokenizer(
                sent.split(),               # 이미 공백으로 분절된 형태소 토큰들
                is_split_into_words=True,   # 공백 분리 토큰 사용
                add_special_tokens=True,    # [CLS], [SEP] 추가
                padding=False,
                truncation=False,
                return_tensors="pt"         # 바로 Tensor로 반환
            )

            input_ids =    encoded["input_ids"].to(device)    # (1, seq_len)
            attention_mask = encoded["attention_mask"].to(device)

            # 3) 모델에 통과 → 문장 임베딩(pooler_output)
            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                # outputs.pooler_output: (1, hidden_size)
                emb = outputs.pooler_output.squeeze(0)  # → (hidden_size,)

            token_ids = input_ids.squeeze(0).tolist()    # Python 리스트로
            embedding  = emb.cpu().tolist()               # Python 리스트로

            # 4) JSONL 형식으로 한 줄에 저장
            #    { "ids": [...], "emb": [...] }
            record = {
                "ids": token_ids,
                "emb": embedding
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ extracted token IDs & embeddings to {out_path}")

if __name__ == "__main__":
    main()
