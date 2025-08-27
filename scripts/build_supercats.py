from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# sklearn (군집/지표)
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score


# ────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────
def l2norm_nd(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

def read_lines(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s

def knn_sparsify(A: np.ndarray, k: int) -> np.ndarray:
    """각 행에서 상위 k개만 남기고 나머지는 0으로 만드는 희소화(대칭 유지)."""
    if k <= 0:
        return A
    K = A.shape[0]
    B = np.zeros_like(A, dtype=np.float32)
    for i in range(K):
        row = A[i]
        if k >= K - 1:
            idx = np.argsort(row)[::-1]
        else:
            idx = np.argpartition(row, -k)[-k:]
        B[i, idx] = row[idx]
    # 대칭화: max(B, B^T)
    return np.maximum(B, B.T)

def balance_score(sizes: np.ndarray) -> float:
    """
    클러스터 크기 균형성 점수(0~1): min/median 및 퀀타일 기반의 간단 지표.
    값이 클수록 균형적.
    """
    if len(sizes) == 0:
        return 0.0
    sizes = np.asarray(sizes, dtype=np.float64)
    if sizes.sum() == 0:
        return 0.0
    p10, p90 = np.percentile(sizes, [10, 90])
    med = np.median(sizes)
    if med <= 0:
        return 0.0
    # 분산이 적고 p10도 너무 작지 않을수록 높게
    score = 0.5 * (p10 / med) + 0.5 * (med / (p90 + 1e-9))
    return float(np.clip(score, 0.0, 1.0))

def eval_grouping(C_norm: np.ndarray, g: np.ndarray) -> Dict[str, float]:
    """
    군집 평가(간단): 코사인 공간에서 실루엣(샘플=centroid), 균형성 혼합 점수.
    반환: {"sil": ..., "bal": ..., "mix": ...}
    """
    # silhouette: sklearn은 거리 기준 → cosine 거리를 쓰려면 입력 벡터는 L2 정규화 상태
    try:
        sil = float(silhouette_score(C_norm, g, metric="cosine"))
    except Exception:
        sil = 0.0
    # balance는 각 그룹 크기 균형
    _, counts = np.unique(g, return_counts=True)
    bal = balance_score(counts)
    # 혼합 점수: 실루엣(가중 0.7) + 균형(0.3)
    mix = 0.7 * sil + 0.3 * bal
    return {"sil": sil, "bal": bal, "mix": mix}

def summarize_super(
    super_id: int,
    members: List[int],
    A: np.ndarray,
    labels: np.ndarray,
    tokens_path: Path,
    topn_tokens: int = 100,
) -> Tuple[int, List[Tuple[str, int]]]:
    """
    슈퍼카테고리 요약: 대표(중심) 클러스터, 상위 토큰 목록 반환.
    대표 클러스터: A 내에서 row-sum이 가장 큰 멤버.
    """
    # 대표(중심) 클러스터
    if len(members) == 1:
        central = members[0]
    else:
        row_sum = [(k, float(A[k, members].sum())) for k in members]
        row_sum.sort(key=lambda x: x[1], reverse=True)
        central = row_sum[0][0]

    # 상위 토큰
    token_ctr = Counter()
    # labels와 tokens.txt의 행 인덱스는 문장 단위로 1:1 매칭되어 있다고 가정
    with tokens_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= len(labels):
                break
            c = int(labels[i])
            if c in members:
                token_ctr.update(line.strip().split())

    top = token_ctr.most_common(topn_tokens)
    return central, top

# ────────────────────────────────────────────────────────────────
# 코어 로직
# ────────────────────────────────────────────────────────────────
def build_affinity(
    C: np.ndarray,
    labels: np.ndarray,
    tokens_path: Path,
    use_cooccur: bool = True,
    alpha: float = 0.7,
    knn_k: int = 10,
) -> np.ndarray:
    """
    최종 affinity A 구성:
      A = alpha * S_cos + (1 - alpha) * S_co   (둘 다 [0,1]로 정규화)
    - S_cos: L2정규화된 센트로이드 코사인 유사도
    - S_co : 문장 단위 공출현 기반 (옵션)
    - 희소화: k-NN sparsify (옵션)
    """
    K = C.shape[0]
    Cn = l2norm_nd(C, axis=1)
    S_cos = (Cn @ Cn.T).astype(np.float32)
    np.fill_diagonal(S_cos, 0.0)
    S_cos = np.clip(S_cos, 0.0, 1.0)

    if use_cooccur:
        Co = np.zeros((K, K), dtype=np.float32)
        # 문장 내 고유 클러스터 페어에 대해 +1
        # tokens.txt는 단순히 라인 수 동기화 용도(빠르게 순회)
        with tokens_path.open(encoding="utf-8") as f:
            for i, _ in enumerate(f):
                if i >= len(labels):
                    break
                # 한 문장에서 등장한 고유 클러스터 집합
                # (이미 labels가 문장→클러스터 id 1개인 경우엔 set의 크기가 1일 수 있음)
                # 만약 토큰 단위 라벨이 아니라 문장 단위 라벨이라면 공출현 효과는 약함.
                ks = {int(labels[i])}
                if len(ks) >= 2:
                    for a, b in combinations(sorted(ks), 2):
                        Co[a, b] += 1.0
                        Co[b, a] += 1.0
        if Co.max() > 0:
            S_co = (Co / Co.max()).astype(np.float32)
        else:
            S_co = Co
    else:
        S_co = np.zeros_like(S_cos, dtype=np.float32)

    A = alpha * S_cos + (1.0 - alpha) * S_co
    A = np.clip(A, 0.0, 1.0)

    if knn_k > 0 and knn_k < K:
        A = knn_sparsify(A, k=knn_k)
    return A

def cluster_super(A: np.ndarray, method: str, M: int, random_state: int = 42) -> np.ndarray:
    K = A.shape[0]
    if method == "agglo":
        # cosine affinity → distance = 1 - A
        D = 1.0 - A
        np.fill_diagonal(D, 0.0)

        # ✅ sklearn 최신: metric='precomputed' 사용
        try:
            mdl = AgglomerativeClustering(
                n_clusters=M,
                metric="precomputed",   # new API
                linkage="average",
                compute_full_tree=False,
            )
            g = mdl.fit_predict(D)
        except TypeError:
            # 🔁 구버전 호환: affinity='precomputed'
            mdl = AgglomerativeClustering(
                n_clusters=M,
                affinity="precomputed",  # old API
                linkage="average",
                compute_full_tree=False,
            )
            g = mdl.fit_predict(D)

        return g.astype(np.int32)

    elif method == "spectral":
        mdl = SpectralClustering(
            n_clusters=M,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
        )
        g = mdl.fit_predict(A)
        return g.astype(np.int32)

    else:
        raise ValueError(f"unknown method: {method}")


# ────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="클러스터 → 슈퍼카테고리 자동 압축")
    ap.add_argument("--centroids", type=Path, required=True, help="centroids.npy (K,H)")
    ap.add_argument("--labels", type=Path, required=True, help="labels.npy (N,)")
    ap.add_argument("--tokens", type=Path, required=True, help="tokens.txt (N lines)")
    ap.add_argument("--out_map", type=Path, required=True, help="출력: super_map.json")
    ap.add_argument("--preview", type=Path, required=True, help="출력: super_preview.txt")
    ap.add_argument("--out_labels", type=Path, default=None, help="(옵션) super_labels.npy")
    ap.add_argument("--alpha", type=float, default=0.7, help="A = α·S_cos + (1-α)·S_co")
    ap.add_argument("--knn_k", type=int, default=10, help="Affinity kNN sparsify (0=off)")
    ap.add_argument("--method", choices=["agglo", "spectral"], default="agglo")
    ap.add_argument("--cands", type=str, default="12,14,16",
                    help="슈퍼카테고리 후보 개수(콤마구분)")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--topn_tokens", type=int, default=120)
    args = ap.parse_args()

    # 0) 로드
    C = np.load(args.centroids)  # (K,H)
    if C.ndim != 2:
        raise ValueError("centroids.npy must be 2D (K,H)")
    K, H = C.shape
    print(f"[load] centroids: {args.centroids}  shape={C.shape}")

    y = np.load(args.labels)     # (N,)
    if y.ndim != 1:
        raise ValueError("labels.npy must be 1D (N,)")
    N = len(y)
    print(f"[load] labels: {args.labels}  shape={y.shape}  (N={N})")

    # 1) Affinity
    A = build_affinity(
        C=C,
        labels=y,
        tokens_path=args.tokens,
        use_cooccur=True,
        alpha=args.alpha,
        knn_k=args.knn_k,
    )
    print(f"[affinity] alpha={args.alpha}  knn_k={args.knn_k}  method={args.method}")

    # 2) 스윕하여 최적 M 선택
    Cn = l2norm_nd(C, axis=1)
    cand_M = [int(s) for s in args.cands.split(",") if s.strip()]
    best = None
    log_rows = []
    for M in cand_M:
        g = cluster_super(A, args.method, M, args.random_state)
        scores = eval_grouping(Cn, g)
        row = {"M": M, **scores}
        log_rows.append(row)
        mix = scores["mix"]
        if (best is None) or (mix > best[0]):
            best = (mix, M, g, scores)
        print(f"[sweep] M={M:2d}  sil={scores['sil']:.4f}  bal={scores['bal']:.4f}  mix={scores['mix']:.4f}")

    # 3) 최종 선택
    assert best is not None
    _, M_best, g_best, best_scores = best
    print(f"[choose] M={M_best}  sil={best_scores['sil']:.4f}  bal={best_scores['bal']:.4f}  mix={best_scores['mix']:.4f}")

    # 4) 저장: super_map.json
    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    super_map = {str(k): int(g_best[k]) for k in range(K)}
    args.out_map.write_text(json.dumps(super_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] super_map -> {args.out_map}")

    # 5) (옵션) 문장 단위 super_labels.npy
    if args.out_labels:
        sup_labels = np.array([g_best[int(c)] for c in y], dtype=np.int32)
        Path(args.out_labels).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out_labels, sup_labels)
        print(f"[save] super_labels -> {args.out_labels}  shape={sup_labels.shape}")

    # 6) 프리뷰 파일
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    with args.preview.open("w", encoding="utf-8") as fw:
        fw.write("=== SUPER CATEGORIES PREVIEW ===\n")
        fw.write(f"centroids: {args.centroids} (K={K}, H={H})\n")
        fw.write(f"labels   : {args.labels} (N={N})\n")
        fw.write(f"tokens   : {args.tokens}\n")
        fw.write(f"alpha={args.alpha}  knn_k={args.knn_k}  method={args.method}\n")
        fw.write("sweep logs:\n")
        for r in log_rows:
            fw.write(f" - M={r['M']:2d}  sil={r['sil']:.4f}  bal={r['bal']:.4f}  mix={r['mix']:.4f}\n")
        fw.write("\n")

        # 그룹별 요약
        for s in range(M_best):
            members = [k for k in range(K) if g_best[k] == s]
            central, top = summarize_super(
                super_id=s,
                members=members,
                A=A,
                labels=y,
                tokens_path=args.tokens,
                topn_tokens=args.topn_tokens,
            )
            fw.write(f"[super {s}]\n")
            fw.write(f" members: {sorted(members)}\n")
            fw.write(f" central: {central}\n")
            # 상위 토큰 40개만 프린트
            top40 = ", ".join([f"{w}:{c}" for w, c in top[:40]])
            fw.write(f" top_tokens: {top40}\n\n")

    print(f"[save] preview -> {args.preview}")


if __name__ == "__main__":
    main()
