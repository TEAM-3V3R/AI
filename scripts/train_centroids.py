import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def l2_normalize(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """row-wise L2 정규화 (X := X / ||X||)"""
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return X / nrm


def choose_model(k: int, mini: bool, max_iter: int, n_init: int, seed: int, batch_size: int):
    if mini:
        return MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=batch_size,
            n_init=n_init,
            max_iter=max_iter,
            init="k-means++",
            verbose=0
        )
    else:
        return KMeans(
            n_clusters=k,
            random_state=seed,
            n_init="auto",
            max_iter=max_iter,
            init="k-means++",
            verbose=0
        )


def eval_scores(X: np.ndarray, labels: np.ndarray, metric: str):
    """품질 지표 계산 (샘플에서만 호출 권장).
       silhouette: metric 지정 / DBI: 유클리드 기반."""
    sil = float(silhouette_score(X, labels, metric=metric))
    dbi = float(davies_bouldin_score(X, labels))  # Euclidean
    return sil, dbi


def try_k(X: np.ndarray, k: int, args) -> tuple:
    """주어진 X(샘플)에 대해 k로 모델 학습 및 지표 산출"""
    model = choose_model(k, args.mini, args.max_iter, args.n_init, args.seed, args.batch_size)
    model.fit(X)
    labels = model.labels_
    metric_for_sil = "cosine" if args.metric == "cosine" else "euclidean"
    sil, dbi = eval_scores(X, labels, metric=metric_for_sil)
    return model, sil, dbi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embs", required=True, help="임베딩 NPY 경로 (예: AI/DPDT/data/embs.npy)")
    ap.add_argument("--k", type=int, required=True, help="기준 K")
    ap.add_argument("--sweep", action="store_true", help="K 스윕 수행 (기준K의 ±20%)")
    ap.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine", help="군집 거리 기준")
    ap.add_argument("--mini", action="store_true", help="MiniBatchKMeans 사용")
    ap.add_argument("--batch_size", type=int, default=4096, help="MiniBatchKMeans 배치 크기")
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=20000, help="스윕/평가용 샘플 수(0이면 전체, 비추)")
    ap.add_argument("--outdir", required=True, help="결과 저장 폴더")
    args = ap.parse_args()

    embs_path = Path(args.embs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 로드
    X = np.load(embs_path, mmap_mode="r")
    N, D = X.shape
    print(f"[load] X: {embs_path} shape={X.shape}, dtype={X.dtype}")

    # 2) 스윕/평가용 샘플 작성
    if args.sample and args.sample < N:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(N, size=args.sample, replace=False)
        X_eval = np.asarray(X[idx], dtype=np.float32, order="C")
        print(f"[sample] use {args.sample} / {N}")
    else:
        # 전체를 평가 대상으로 쓰지 않도록 경고만 남기고 그대로 사용 가능
        X_eval = np.asarray(X, dtype=np.float32, order="C")
        if X_eval.shape[0] > 50000:
            print("[warn] Evaluation on full data is expensive; consider --sample to speed up.")

    # 3) metric=cosine이면 평가 데이터 정규화
    if args.metric == "cosine":
        X_eval = l2_normalize(X_eval)

    # 4) 후보 K 집합
    if args.sweep:
        ks = sorted(set([args.k] + [max(2, int(args.k * r)) for r in (0.8, 0.9, 1.1, 1.2)]))
    else:
        ks = [args.k]

    # 5) 샘플로 스윕
    results = []
    best = None
    print(f"[train] Ks={ks}  metric={args.metric}  mini={args.mini}")
    for kk in ks:
        mdl, sil, dbi = try_k(X_eval, kk, args)
        results.append({"k": kk, "silhouette": sil, "dbi": dbi})
        print(f" - k={kk:4d}  silhouette={sil:.4f}  dbi={dbi:.4f}")
        if best is None or sil > best[2]:
            best = (kk, mdl, sil, dbi)

    best_k, best_model, best_sil_eval, best_dbi_eval = best

    # 6) 전체 데이터로 최종 학습 (refit)
    print("[refit] re-fitting best K on full data…")
    X_full = np.asarray(X, dtype=np.float32, order="C")
    if args.metric == "cosine":
        X_full = l2_normalize(X_full)

    final_model = choose_model(best_k, args.mini, args.max_iter, args.n_init, args.seed, args.batch_size)
    final_model.fit(X_full)
    labels_full = final_model.labels_

    # 7) refit 후 메트릭은 '샘플'에서만 다시 평가 (전체 O(N^2) 방지)
    if args.sample and args.sample < N:
        # 샘플 인덱스 재사용을 위해 동일 시드로 재추출
        rng = np.random.default_rng(args.seed)
        ss = min(args.sample, N)
        idx2 = rng.choice(N, size=ss, replace=False)
        X_eval2 = np.asarray(X[idx2], dtype=np.float32, order="C")
        if args.metric == "cosine":
            X_eval2 = l2_normalize(X_eval2)
        labels_eval2 = labels_full[idx2]
        sil_refit, dbi_refit = eval_scores(X_eval2, labels_eval2,
                                           metric=("cosine" if args.metric == "cosine" else "euclidean"))
    else:
        sil_refit, dbi_refit = float("nan"), float("nan")

    # 8) 저장
    np.save(outdir / "centroids.npy", final_model.cluster_centers_.astype(np.float32))
    np.save(outdir / "labels.npy", labels_full.astype(np.int32))
    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "candidates": results,                     # 샘플 스윕 성적표
                "best": {                                  # 스윕에서 고른 K (샘플 기준)
                    "k": int(best_k),
                    "silhouette_eval": float(best_sil_eval),
                    "dbi_eval": float(best_dbi_eval)
                },
                "refit_eval": {                            # 전체 refit 후, '샘플' 재평가
                    "silhouette_eval": float(sil_refit),
                    "dbi_eval": float(dbi_refit)
                },
                "config": {
                    "metric": args.metric,
                    "mini": args.mini,
                    "batch_size": args.batch_size,
                    "max_iter": args.max_iter,
                    "n_init": args.n_init,
                    "seed": args.seed,
                    "sample": int(args.sample)
                }
            },
            f, ensure_ascii=False, indent=2
        )

    print(f"[best] k={best_k}  (eval sil={best_sil_eval:.4f}, dbi={best_dbi_eval:.4f})")
    print(f"[refit-eval] sil={sil_refit:.4f}, dbi={dbi_refit:.4f}  (on sample)")
    print(f"[save] {outdir/'centroids.npy'}")
    print(f"[save] {outdir/'labels.npy'}")
    print(f"[save] {outdir/'metrics.json'}")


if __name__ == "__main__":
    main()
