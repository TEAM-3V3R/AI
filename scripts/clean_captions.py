import argparse, hashlib, os, re, unicodedata, random
from pathlib import Path
from collections import defaultdict

# ----- 옵션성: Mecab 있으면 사용, 없으면 fallback -----
try:
    from konlpy.tag import Mecab
    mecab = Mecab()
except Exception:
    mecab = None

POS_KEEP = {"NNG","NNP","VA","VV","XR"}  # 명사/형용사/동사 계열
STOPWORDS = set("""
정면 좌측 우측 시선 향해 카메라 포즈 무표정 기쁜 슬픈 분노한 청년 중년 여성 남성 소년 소녀 사람 얼굴 표정 바라보다 응시하다
""".split())

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def tokens(s: str):
    if mecab:
        out = []
        for w,p in mecab.pos(s):
            if p in POS_KEEP and w not in STOPWORDS:
                out.append(w.lower())
        return out
    return [t for t in re.sub(r"[^0-9A-Za-z가-힣 ]"," ", s).split() if t not in STOPWORDS]

def ngrams(seq, n=3):
    return set(tuple(seq[i:i+n]) for i in range(max(0, len(seq)-n+1)))

def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter/union if union else 0.0

def signature_key(tok):
    """샤딩용 서명: 토큰 상위 6개를 정렬해 결합"""
    key_terms = tuple(sorted(tok[:6]))
    raw = "|".join(key_terms).encode("utf-8")
    return hashlib.md5(raw).hexdigest()

def sha_idx(sig_hex: str, shards: int) -> int:
    return int(sig_hex[:8], 16) % shards

def template_key(tok):
    """패턴 억제용 러프 키"""
    key_terms = [w for w in tok if w not in STOPWORDS][:6]
    return tuple(key_terms[:4])

def streaming_exact_dedup(inp: Path, tmp_exact: Path):
    seen = set()
    n_in = n_out = 0
    with inp.open("r", encoding="utf-8") as fi, tmp_exact.open("w", encoding="utf-8") as fo:
        for line in fi:
            n_in += 1
            s = normalize(line)
            if not s: continue
            h = hashlib.md5(s.encode("utf-8")).digest()
            if h in seen: continue
            seen.add(h)
            fo.write(s + "\n")
            n_out += 1
    return n_in, n_out

def shard_by_signature(tmp_exact: Path, tmp_dir: Path, shards: int):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fps = [ (tmp_dir / f"shard_{i:04d}.txt").open("w", encoding="utf-8") for i in range(shards) ]
    n_in = 0
    with tmp_exact.open("r", encoding="utf-8") as f:
        for line in f:
            n_in += 1
            s = line.rstrip("\n")
            tok = tokens(s)
            sig = signature_key(tok)
            idx = sha_idx(sig, shards)
            fps[idx].write(s + "\n")
    for fp in fps: fp.close()
    return n_in

def process_shard_file(shard_fp: Path, out: Path, jacc_th: float, max_per_template: int):
    """샤드 내부에서 근접중복 제거 + 패턴 억제 → out에 append"""
    if shard_fp.stat().st_size == 0:
        return 0,0
    lines = [normalize(s) for s in shard_fp.read_text(encoding="utf-8").splitlines() if s.strip()]
    # exact-dedup within shard (just in case)
    lines = list(dict.fromkeys(lines))
    toks  = [tokens(s) for s in lines]
    grams = [ngrams(t,3) for t in toks]

    keep = []
    kept_idx = []
    for i,g in enumerate(grams):
        dup = False
        # 샤드라서 수천 라인 규모 → 전수 비교해도 OK.
        for j in kept_idx:
            if jaccard(g, grams[j]) >= jacc_th:
                dup = True; break
        if not dup:
            kept_idx.append(i); keep.append(True)
        else:
            keep.append(False)
    lines = [s for s,k in zip(lines, keep) if k]
    toks  = [t for t,k in zip(toks,  keep) if k]

    # 패턴 억제
    buckets = defaultdict(list)
    for i,(s,t) in enumerate(zip(lines, toks)):
        buckets[template_key(t)].append(s)
    balanced = []
    for _, arr in buckets.items():
        random.shuffle(arr)
        if max_per_template < 999999:
            balanced.extend(arr[:max_per_template])
        else:
            balanced.extend(arr)
    # 저장(append)
    with out.open("a", encoding="utf-8") as fo:
        for s in balanced:
            fo.write(s + "\n")
    return len(keep), len(balanced)

def main(inp: Path, out: Path, workdir: Path, shards: int, jacc_th: float, max_per_template: int):
    workdir.mkdir(parents=True, exist_ok=True)
    tmp_exact = workdir / "exact.tmp"
    shard_dir = workdir / "shards"
    if out.exists(): out.unlink()  # 새로 생성

    print("[1/3] exact dedup streaming...")
    n_in, n_exact = streaming_exact_dedup(inp, tmp_exact)
    print(f"  input lines   : {n_in:,}")
    print(f"  unique (exact): {n_exact:,} -> {tmp_exact}")

    print("[2/3] sharding by signature...")
    n_shard_in = shard_by_signature(tmp_exact, shard_dir, shards)
    print(f"  sharded lines : {n_shard_in:,} into {shards} shard files")

    print("[3/3] per-shard near-dup + pattern limiting...")
    total_before = total_after = 0
    for i, shard_fp in enumerate(sorted(shard_dir.glob("shard_*.txt"))):
        b, a = process_shard_file(shard_fp, out, jacc_th=jacc_th, max_per_template=max_per_template)
        total_before += b; total_after += a
        if (i+1) % max(1, shards//10) == 0:
            print(f"  processed {i+1}/{shards} shards")

    # 정리(선택) : 임시파일 삭제
    try:
        tmp_exact.unlink()
        for fp in shard_dir.glob("shard_*.txt"):
            fp.unlink()
        shard_dir.rmdir()
    except Exception:
        pass

    print("[done]")
    print(f"  near-dup step in : {total_before:,}")
    print(f"  final written    : {total_after:,} -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True, help="입력 txt (아주 큰 파일)")
    ap.add_argument("--out", type=Path, required=True, help="최종 결과 txt")
    ap.add_argument("--workdir", type=Path, default=Path("work/clean_large"), help="중간 산출물 폴더")
    ap.add_argument("--shards", type=int, default=1024, help="샤드 개수(유사 문장끼리 분산)")
    ap.add_argument("--jacc_th", type=float, default=0.90, help="3-gram Jaccard 임계값 (0.85~0.95)")
    ap.add_argument("--max_per_template", type=int, default=9999, help="템플릿 그룹당 최대 유지 개수")
    args = ap.parse_args()

    main(args.inp, args.out, args.workdir, args.shards, args.jacc_th, args.max_per_template)
