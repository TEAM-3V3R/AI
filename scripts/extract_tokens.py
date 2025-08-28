import argparse
import json
from pathlib import Path
from typing import List, Dict, Union

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ------------------------- pooling -------------------------
def pool_hidden_states(last_hidden_state: torch.Tensor,
                       attention_mask: torch.Tensor,
                       how: str = "mean") -> torch.Tensor:
    """
    last_hidden_state: (B, L, H)
    attention_mask:    (B, L)
    return: (B, H)
    """
    if how == "cls":
        return last_hidden_state[:, 0]  # [CLS]
    mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
    masked = last_hidden_state * mask

    if how == "mean":
        denom = mask.sum(dim=1).clamp(min=1)
        return masked.sum(dim=1) / denom
    elif how == "max":
        masked = masked + (1 - mask) * (-1e9)  # mask=0 위치는 -inf
        return masked.max(dim=1).values
    else:
        raise ValueError(f"Unknown pooling: {how}")


# --------------------- windowing encode ---------------------
def window_encode(tokenizer: AutoTokenizer,
                  tokens_or_text: Union[List[str], str],
                  split_tokens: bool,
                  max_len: int,
                  stride: int):
    """
    긴 입력을 여러 윈도우로 나눠 인코딩 dict 들을 리스트로 반환
    각 dict는 "input_ids"(1,L), "attention_mask"(1,L)를 가짐
    """
    encs = []

    if split_tokens:
        toks = tokens_or_text if isinstance(tokens_or_text, list) else str(tokens_or_text).split()
        # [CLS], [SEP] 포함해야 하므로 내부 길이는 (max_len - 2)
        inner_max = max_len - 2
        if inner_max <= 0:
            inner_max = 2  # 안전장치

        # stride는 겹치는 토큰 수. step은 이동 폭.
        step = inner_max - stride
        if step <= 0:
            step = inner_max

        i = 0
        while i < len(toks):
            sub = toks[i:i + inner_max]
            enc = tokenizer(
                sub,
                is_split_into_words=True,
                add_special_tokens=True,
                padding=False,
                truncation=True,
                return_tensors="pt",
                max_length=max_len,
            )
            encs.append(enc)
            if len(toks) <= inner_max:
                break
            i += step
    else:
        text = tokens_or_text if isinstance(tokens_or_text, str) else " ".join(tokens_or_text)
        # 토크나이저로 raw ids를 먼저 얻어 길이를 판단
        ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids) <= max_len:
            enc = tokenizer(
                text,
                add_special_tokens=True,
                padding=False,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            encs.append(enc)
        else:
            # [CLS] + chunk + [SEP] 구조로 수동 슬라이싱
            inner = ids[1:-1]  # [CLS], [SEP] 제거한 본문
            inner_max = max_len - 2
            if inner_max <= 0:
                inner_max = 2
            step = inner_max - stride
            if step <= 0:
                step = inner_max

            cls_id = tokenizer.cls_token_id
            sep_id = tokenizer.sep_token_id

            i = 0
            while i < len(inner):
                piece = inner[i:i + inner_max]
                chunk_ids = [cls_id] + piece + [sep_id]
                ids_tensor = torch.tensor([chunk_ids], dtype=torch.long)
                am_tensor = torch.ones_like(ids_tensor)
                encs.append({"input_ids": ids_tensor, "attention_mask": am_tensor})
                i += step

    return encs


def main():
    ap = argparse.ArgumentParser(
        description="(형태소 공백분리 or 자연문장) → token IDs + sentence embeddings(JSONL)"
    )
    ap.add_argument("-i", "--input", required=True,
                    help="예: AI/DPDT/data/tokens.txt 또는 corpus_filtered.txt")
    ap.add_argument("-o", "--output", required=True,
                    help="예: AI/DPDT/data/tokens_and_embs.jsonl")
    ap.add_argument("-m", "--model", default="skt/kobert-base-v1",
                    help="예: skt/kobert-base-v1")

    # kobert 호환을 위해 기본 False. 필요시 --use_fast 로 True.
    ap.add_argument("--use_fast", action="store_true", default=False,
                    help="Fast tokenizer 사용 (기본 False; kobert 권장 False)")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--pooling", choices=["cls", "mean", "max"], default="mean")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--fp16", action="store_true",
                    help="CUDA 사용 시 half precision으로 추론")
    ap.add_argument("--split_tokens", action="store_true", default=True,
                    help="입력이 공백 분리 형태소일 때 사용 (기본 True)")
    ap.add_argument("--no_split_tokens", action="store_true",
                    help="자연문장 입력일 때 지정 (split_tokens=False)")

    args = ap.parse_args()

    # split_tokens 최종 결정
    if args.no_split_tokens:
        split_tokens = False
    else:
        split_tokens = args.split_tokens

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"⏳ Loading tokenizer/model: {args.model} (use_fast={args.use_fast})")
    # kobert는 fast 변환에서 문제 많음 → 기본 False. 옵션으로 강제 on 가능.
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=args.use_fast)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.fp16 and device.type == "cuda":
        model.half()

    # 라인 수(진행바)
    try:
        n_lines = sum(1 for _ in inp.open(encoding="utf-8"))
    except Exception:
        n_lines = None

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        pbar = tqdm(total=n_lines, unit="line", disable=(n_lines is None))

        buf_ids: List[List[int]] = []
        buf_am:  List[List[int]] = []
        buf_meta: List[Dict] = []

        def flush_batch():
            if not buf_meta:
                return
            # pad to same length
            ids_tensors = [torch.tensor(x, dtype=torch.long) for x in buf_ids]
            am_tensors  = [torch.tensor(x, dtype=torch.long) for x in buf_am]
            input_ids = torch.nn.utils.rnn.pad_sequence(ids_tensors, batch_first=True, padding_value=pad_id)
            attention_mask = torch.nn.utils.rnn.pad_sequence(am_tensors, batch_first=True, padding_value=0)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.inference_mode():
                outp = model(input_ids=input_ids, attention_mask=attention_mask)
                # 일부 모델은 pooler_output이 없을 수 있으므로 last_hidden_state 풀링을 기본으로
                last_h = outp.last_hidden_state  # (B, L, H)
                emb = pool_hidden_states(last_h, attention_mask, how=args.pooling)  # (B, H)

            # 저장
            for meta, ids_row, e_row in zip(buf_meta, input_ids, emb):
                record = {
                    "ids": ids_row.tolist(),
                    "emb": e_row.detach().float().cpu().tolist(),
                    "meta": meta
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            buf_ids.clear()
            buf_am.clear()
            buf_meta.clear()

        line_no = 0
        for raw in fin:
            line_no += 1
            sent = raw.strip()
            if not sent:
                pbar.update(1)
                continue

            # 형태소 공백분리 or 자연문장 처리
            payload = sent.split() if split_tokens else sent

            # 길면 윈도우로 나눠 여러 encs 생성
            encs = window_encode(tokenizer, payload, split_tokens, args.max_len, args.stride)

            for win_idx, enc in enumerate(encs):
                ids = enc["input_ids"].squeeze(0).tolist()
                am  = enc["attention_mask"].squeeze(0).tolist()
                buf_ids.append(ids)
                buf_am.append(am)
                buf_meta.append({"line": line_no, "win": win_idx})

                if len(buf_meta) >= args.batch_size:
                    flush_batch()

            pbar.update(1)

        flush_batch()
        pbar.close()

    print(f"✅ Saved: {out} (pooling={args.pooling}, max_len={args.max_len}, stride={args.stride}, split_tokens={split_tokens})")


if __name__ == "__main__":
    main()
