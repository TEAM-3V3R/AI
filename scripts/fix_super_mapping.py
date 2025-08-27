# scripts/fix_super_mapping.py
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

def load_super_map(p: Path):
    m = json.loads(p.read_text(encoding="utf-8"))
    # 허용 포맷: {"0": 3, "1": 7, ...}  또는  ["3","7",...]
    if isinstance(m, dict):
        # 키/값을 int로 강제
        out = {}
        for k, v in m.items():
            out[int(k)] = int(v)
        return out
    elif isinstance(m, list):
        return {i: int(v) for i, v in enumerate(m)}
    else:
        raise ValueError("super_map.json 형식이 올바르지 않습니다.")

def top_keywords(tokens_path: Path, labels: np.ndarray, cl2super: dict,
                 topn_per_super=30, stop=set()):
    """
    tokens.txt와 labels.npy를 이용해 슈퍼카테고리별 대표 키워드 상위 N 추출
    """
    super_counts = defaultdict(Counter)

    with tokens_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= len(labels):
                break
            cid = int(labels[i])
            sid = cl2super.get(cid, None)
            if sid is None:
                continue
            toks = [t for t in line.strip().split() if t and (t not in stop)]
            super_counts[sid].update(toks)

    # 상위 키워드 반환
    return {int(sid): [w for w, _ in cnt.most_common(topn_per_super)]
            for sid, cnt in super_counts.items()}

def main():
    ap = argparse.ArgumentParser(
        description="super_map.json 기반 카테고리 매핑 고정 및 요약 아티팩트 생성"
    )
    ap.add_argument("--centroids", required=True, type=Path,
                    help="models/kmeans_k100/centroids.npy")
    ap.add_argument("--labels", required=True, type=Path,
                    help="models/kmeans_k100/labels.npy  (토큰별 클러스터 ID)")
    ap.add_argument("--tokens", required=True, type=Path,
                    help="data/tokens.txt (전처리 토큰 문장)")
    ap.add_argument("--super_map", required=True, type=Path,
                    help="data/super_map.json (cluster->super)")
    ap.add_argument("--outdir", required=False, type=Path,
                    default=Path("AI/DPDT/data"))
    ap.add_argument("--names", required=False, type=Path,
                    help="(선택) 슈퍼카테고리 이름 JSON, 예: {\"0\":\"생활/실내\", ...}")
    ap.add_argument("--topn", type=int, default=30, help="슈퍼카테고리별 대표 키워드 개수")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1) 로드
    centroids = np.load(args.centroids)           # (K, D)
    labels    = np.load(args.labels)              # (N,)
    cl2super  = load_super_map(args.super_map)    # dict[int->int]

    K = centroids.shape[0]
    if sorted(cl2super.keys()) != list(range(K)):
        raise ValueError(f"super_map의 키(클러스터ID)가 0..{K-1} 범위를 모두 포함해야 합니다.")

    # 2) 슈퍼 이름 불러오거나 자동생성
    if args.names and args.names.exists():
        super_names = {int(k): v for k, v in
                       json.loads(args.names.read_text(encoding="utf-8")).items()}
    else:
        # 자동 이름: super_0, super_1 ...
        super_names = {int(s): f"super_{s}" for s in set(cl2super.values())}

    # 3) 대표 키워드 추출 (불용 키워드 간단 억제)
    basic_stop = {"것", "수", "들", "하다", "되다", "있다", "없다", "이다", "위해", "그리고",
                  "등", "및", "또는", "같다", "같은", "이번", "오늘"}
    super_kws = top_keywords(args.tokens, labels, cl2super,
                             topn_per_super=args.topn, stop=basic_stop)

    # 4) 파일로 저장
    centroid_to_super = [int(cl2super[cid]) for cid in range(K)]

    (args.outdir / "centroid_to_super.json").write_text(
        json.dumps({str(i): int(sid) for i, sid in enumerate(centroid_to_super)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    (args.outdir / "super_names.json").write_text(
        json.dumps({str(k): v for k, v in sorted(super_names.items())},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    (args.outdir / "super_keywords.json").write_text(
        json.dumps({str(k): v for k, v in sorted(super_kws.items())},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 런타임 패키지 묶음
    bundle = {
        "centroids_path": str(args.centroids).replace("\\", "/"),
        "centroid_to_super": centroid_to_super,   # 리스트(정수)
        "super_names": {str(k): v for k, v in super_names.items()},
        "super_keywords": {str(k): v for k, v in super_kws.items()},
    }
    (args.outdir / "runtime_categories.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("[done]")
    print(" - centroid_to_super.json")
    print(" - super_names.json")
    print(" - super_keywords.json")
    print(" - runtime_categories.json")

if __name__ == "__main__":
    main()
