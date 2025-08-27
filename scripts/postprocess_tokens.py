import argparse, collections, sys
from pathlib import Path

def count_unigrams(inp: Path):
    cnt = collections.Counter()
    with inp.open(encoding="utf-8") as f:
        for line in f:
            for tok in line.split():
                cnt[tok] += 1
    return cnt

def count_bigrams(inp: Path, keep_unigram, sep="_"):
    cnt = collections.Counter()
    with inp.open(encoding="utf-8") as f:
        for line in f:
            toks = [t for t in line.split() if t in keep_unigram]
            for i in range(len(toks) - 1):
                bg = f"{toks[i]}{sep}{toks[i+1]}"
                cnt[bg] += 1
    return cnt

def main(inp: Path, out: Path,
         min_freq=3, min_bigram=20, sep="_",
         top_k_common=0, min_tokens=1):
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) unigram 카운트 & 희귀 제거 집합
    ucnt = count_unigrams(inp)
    keep_uni = {t for t, c in ucnt.items() if c >= min_freq}

    # (선택) 너무 흔한 공용어 top-k 제거
    if top_k_common > 0 and len(keep_uni) > top_k_common:
        common = {t for t, _ in ucnt.most_common(top_k_common)}
        keep_uni -= common

    # 2) bigram 카운트 (희귀 제거 후 토큰으로 계산)
    bcnt = count_bigrams(inp, keep_uni, sep=sep)
    keep_bi = {b for b, c in bcnt.items() if c >= min_bigram}

    # 3) 최종 쓰기
    n_in, n_out = 0, 0
    with inp.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
        for line in f:
            n_in += 1
            toks = [t for t in line.split() if t in keep_uni]
            bigrams = []
            for i in range(len(toks) - 1):
                bg = f"{toks[i]}{sep}{toks[i+1]}"
                if bg in keep_bi:
                    bigrams.append(bg)

            final = toks + bigrams
            if len(final) >= min_tokens:
                w.write(" ".join(final) + "\n")
                n_out += 1

    # 로그
    print(f"[done] lines: {n_in} -> {n_out}")
    print(f"       vocab(unigram kept) = {len(keep_uni)}  (min_freq >= {min_freq}"
          + (f", top-{top_k_common} common removed" if top_k_common else "") + ")")
    print(f"       vocab(bigram kept)  = {len(keep_bi)}  (min_bigram >= {min_bigram})")
    print(f"       out -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp",  type=Path, required=True,
                    help="입력 파일 (예: DPDT/data/corpus_filtered.txt)")
    ap.add_argument("--out",  type=Path, required=True,
                    help="출력 파일 (예: DPDT/data/tokens.txt)")
    ap.add_argument("--min_freq",    type=int, default=3)
    ap.add_argument("--min_bigram",  type=int, default=20)
    ap.add_argument("--sep",         type=str, default="_")
    ap.add_argument("--top_k_common", type=int, default=0,
                    help="상위 공용어 N개 제거 (0=비활성)")
    ap.add_argument("--min_tokens",  type=int, default=1,
                    help="최소 토큰수 미만 라인은 제거")
    args = ap.parse_args()

    main(args.inp, args.out,
         min_freq=args.min_freq,
         min_bigram=args.min_bigram,
         sep=args.sep,
         top_k_common=args.top_k_common,
         min_tokens=args.min_tokens)
