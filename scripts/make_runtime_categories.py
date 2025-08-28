import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np


def load_centroid_to_super(super_map_path: Path) -> dict[int, int]:
    """
    super_map.json(dict or list) 또는 centroid_to_super.json(dict) 를 받아
    {cluster_id -> super_id} dict로 표준화해서 반환.
    """
    data = json.loads(super_map_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # {"0": 3, "1": 7, ...} or {"0": "3", ...}
        return {int(k): int(v) for k, v in data.items()}
    elif isinstance(data, list):
        # ["3", "7", ...] or [3, 7, ...]
        return {i: int(v) for i, v in enumerate(data)}
    else:
        raise ValueError("super_map / centroid_to_super JSON 형식을 알 수 없습니다.")


def build_super_keywords(tokens_path: Path, labels: np.ndarray,
                         cl2super: dict[int, int], topn: int = 30) -> dict[int, list[str]]:
    """
    tokens.txt(한 줄 1문장, 공백분리 토큰) + labels.npy(문장→클러스터)
    → 슈퍼카테고리별 상위 키워드 추출
    """
    stop_basic = {"것", "수", "들", "하다", "되다", "있다", "없다", "이다",
                  "그리고", "등", "및", "또는", "이번", "오늘", "같다", "같은"}
    sup_ctr = defaultdict(Counter)

    with tokens_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= len(labels):
                break
            cid = int(labels[i])
            sid = cl2super.get(cid)
            if sid is None:
                continue
            toks = [t for t in line.strip().split() if t and t not in stop_basic]
            if toks:
                sup_ctr[sid].update(toks)

    return {int(s): [w for w, _ in ctr.most_common(topn)] for s, ctr in sup_ctr.items()}


def main():
    ap = argparse.ArgumentParser(description="runtime_categories.json 자동 생성")
    ap.add_argument("--centroids", type=Path, required=True, help=".../centroids.npy")
    ap.add_argument("--super_map", type=Path, required=True,
                    help=".../super_map.json 또는 .../centroid_to_super.json")
    ap.add_argument("--labels", type=Path, help="(옵션) .../labels.npy")
    ap.add_argument("--tokens", type=Path, help="(옵션) .../tokens.txt")
    ap.add_argument("--names",  type=Path, help="(옵션) 슈퍼카테고리 이름 JSON {\"0\":\"생활/실내\", ...}")
    ap.add_argument("--out",    type=Path, default=Path("AI/DPDT/data/runtime_categories.json"))
    ap.add_argument("--topn_keywords", type=int, default=30)
    args = ap.parse_args()

    # centroids 로드 → K 확인
    centroids = np.load(args.centroids)
    if centroids.ndim != 2:
        raise ValueError("centroids.npy는 (K, D) 2차원 배열이어야 합니다.")
    K = centroids.shape[0]

    # cluster -> super dict
    cl2super = load_centroid_to_super(args.super_map)
    expected_keys = list(range(K))
    if sorted(cl2super.keys()) != expected_keys:
        raise ValueError(f"super_map의 키가 0..{K-1} 전체를 포함해야 합니다. (현재 키: {sorted(cl2super.keys())[:5]}...)")

    # names 불러오기(없으면 super_# 자동)
    if args.names and args.names.exists():
        super_names = {int(k): v for k, v in
                       json.loads(args.names.read_text(encoding="utf-8")).items()}
    else:
        super_names = {int(s): f"super_{s}" for s in sorted(set(cl2super.values()))}

    # keywords 생성(옵션)
    super_keywords = {}
    if args.labels and args.tokens and args.labels.exists() and args.tokens.exists():
        labels = np.load(args.labels)
        if labels.ndim != 1:
            raise ValueError("labels.npy는 1차원 배열이어야 합니다.")
        super_keywords = build_super_keywords(args.tokens, labels, cl2super, args.topn_keywords)

    # 리스트 형태로도 제공 (로딩 성능/간결성)
    centroid_to_super = [int(cl2super[i]) for i in range(K)]

    bundle = {
        "centroids_path": str(args.centroids).replace("\\", "/"),
        "centroid_to_super": centroid_to_super,
        "super_names": {str(k): v for k, v in super_names.items()},
        "super_keywords": {str(k): v for k, v in super_keywords.items()}
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] runtime_categories.json ->", args.out)


if __name__ == "__main__":
    main()
