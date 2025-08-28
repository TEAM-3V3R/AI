# AI/scripts/sanity_check.py
import argparse, json
from pathlib import Path
from collections import Counter

import numpy as np
from numpy.linalg import norm

# sklearn은 있으면 사용(실루엣/DBI), 없으면 건너뜀
try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    HAS_SK = True
except Exception:
    HAS_SK = False


def pstats(arr, ps=(5, 10, 25, 50, 75, 90, 95)):
    arr = np.asarray(arr)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def check_files(centroids, labels, metrics, tokens):
    print("=== [FILES] 존재 여부 ===")
    for k, p in {
        "centroids": centroids,
        "labels": labels,
        "metrics": metrics,
        "tokens": tokens,
    }.items():
        if p:
            print(f"- {k:9s}:", Path(p).exists(), "->", p)
    print()


def load_centroids(path):
    C = np.load(path)
    print("=== [CENTROIDS] ===")
    print("shape:", C.shape, "dtype:", C.dtype)
    cn = norm(C, axis=1)
    print("norm mean/std/min/max:", float(cn.mean()), float(cn.std()), float(cn.min()), float(cn.max()))
    z = int((cn < 1e-8).sum())
    if z:
        print(f"WARNING: zero-norm centroids: {z}")
    # 코사인 중복(매우 유사) 검사
    Cn = C / (cn[:, None] + 1e-9)
    K = C.shape[0]
    dup = []
    if K <= 200:  # 너무 크면 생략
        S = Cn @ Cn.T
        np.fill_diagonal(S, -1.0)
        i, j = np.where(S > 0.995)  # 0.995 이상이면 거의 중복
        for a, b in zip(i, j):
            if a < b:
                dup.append((int(a), int(b), float(S[a, b])))
    if dup:
        print(f"WARNING: near-duplicate centroids (cos>0.995): {len(dup)}  → 예: {dup[:5]}")
    print()
    return C


def check_labels(labels_path, tokens_path=None):
    y = np.load(labels_path)
    print("=== [LABELS] ===")
    print("labels shape:", y.shape, y.dtype)
    uniq, cnt = np.unique(y, return_counts=True)
    print("K (from labels):", len(uniq))
    print("size min/median/max:", int(cnt.min()), int(np.median(cnt)), int(cnt.max()))
    print("clusters with <= 5 samples:", int((cnt <= 5).sum()))
    print("top10 biggest:", sorted(cnt, reverse=True)[:10])
    if tokens_path and Path(tokens_path).exists():
        n_lines = sum(1 for _ in open(tokens_path, encoding="utf-8"))
        if n_lines != len(y):
            print(f"WARNING: tokens lines({n_lines}) != labels({len(y)})")
    print()
    return y


def check_metrics(metrics_path):
    p = Path(metrics_path)
    if not p.exists():
        print("=== [METRICS] 없음 ===\n")
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    print("=== [METRICS] ===")
    if "candidates" in m:
        best = m.get("best", {})
        print("candidates:", m["candidates"][:3], " ...")
        print("best:", best)
    if "refit_eval" in m:
        print("refit_eval:", m["refit_eval"])
    print()
    return m


def sample_similarity(C, embs_path, sample=10000):
    if not embs_path:
        return
    X = np.load(embs_path, mmap_mode="r")
    N = len(X)
    ss = min(sample, N)
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=ss, replace=False)
    Xs = np.asarray(X[idx], dtype=np.float32)
    # 코사인 유사도 분포
    Cn = C / (norm(C, axis=1, keepdims=True) + 1e-9)
    Xn = Xs / (norm(Xs, axis=1, keepdims=True) + 1e-9)
    sims = Xn @ Cn.T  # (ss, K)
    mx = sims.max(axis=1)
    print("=== [NEAREST-COSINE] 샘플 분포 ===")
    stats = pstats(mx, ps=(5, 10, 25, 50, 75, 90, 95, 99))
    print("max cosine percentiles:", stats)
    # 임계치 제안
    thr = stats["p10"]
    print(f"→ 제안 임계치(threshold) ~ p10 ≈ {thr:.3f} (이보다 낮으면 '기타/리랭크')")
    print()
    return stats


def sample_quality_with_sklearn(C, embs_path, labels_path, sample=5000):
    if not (HAS_SK and embs_path and labels_path):
        return
    X = np.load(embs_path, mmap_mode="r")
    y = np.load(labels_path)
    N = min(len(X), len(y))
    ss = min(sample, N)
    rng = np.random.default_rng(123)
    idx = rng.choice(N, size=ss, replace=False)
    Xs = np.asarray(X[idx], dtype=np.float32)
    ys = y[idx]
    # 코사인 기준 실루엣 (행 정규화)
    Xn = Xs / (norm(Xs, axis=1, keepdims=True) + 1e-9)
    sil = float(silhouette_score(Xn, ys, metric="cosine"))
    dbi = float(davies_bouldin_score(Xn, ys))
    print("=== [SKLEARN SAMPLE QUALITY] ===")
    print(f"silhouette(cosine, n={ss}): {sil:.4f}  |  DBI(euclid): {dbi:.4f}")
    print("(참고: 값 자체보다 '재학습 전후 비교'에 쓰는 지표)")
    print()
    return sil, dbi


def top_tokens_per_cluster(tokens_path, labels_path, topn=15, show_k=5):
    tp = Path(tokens_path)
    lp = Path(labels_path)
    if not (tp.exists() and lp.exists()):
        return
    y = np.load(labels_path)
    cnts = Counter(y.tolist())
    topk = [k for k, _ in cnts.most_common(show_k)]
    buckets = {k: Counter() for k in topk}
    with open(tokens_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= len(y): break
            k = int(y[i])
            if k in buckets:
                buckets[k].update(line.strip().split())
    print("=== [TOP TOKENS (상위 클러스터)] ===")
    for k in topk:
        toks = [w for w, _ in buckets[k].most_common(topn)]
        print(f"cluster {k:3d}: {', '.join(toks)}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centroids", required=True)
    ap.add_argument("--labels", required=False)
    ap.add_argument("--metrics", required=False)
    ap.add_argument("--embs", required=False, help="embs.npy (샘플 유사도 분포/품질 체크용)")
    ap.add_argument("--tokens", required=False, help="tokens.txt (상위 토큰 프린트용)")
    ap.add_argument("--sample", type=int, default=10000, help="샘플 크기(유사도/품질)")
    ap.add_argument("--topn", type=int, default=15)
    args = ap.parse_args()

    check_files(args.centroids, args.labels, args.metrics, args.tokens)
    C = load_centroids(args.centroids)

    if args.metrics:
        check_metrics(args.metrics)
    if args.labels:
        check_labels(args.labels, args.tokens)
    if args.embs:
        sample_similarity(C, args.embs, sample=args.sample)
        sample_quality_with_sklearn(C, args.embs, args.labels, sample=min(5000, args.sample))
    if args.tokens and args.labels:
        top_tokens_per_cluster(args.tokens, args.labels, topn=args.topn, show_k=5)


if __name__ == "__main__":
    main()
