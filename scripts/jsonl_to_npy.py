# AI/scripts/jsonl_to_npy.py
import argparse, json
from pathlib import Path
import numpy as np

def main(inp: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    embs = []
    with inp.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            embs.append(rec["emb"])
            if i % 100000 == 0:
                print(f"[read] {i} lines")

    arr = np.array(embs, dtype=np.float32)
    np.save(out, arr)
    print(f"[done] {inp} -> {out}  shape={arr.shape} dtype={arr.dtype}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="tokens_and_embs.jsonl")
    ap.add_argument("-o", "--output", required=True, help="embs.npy")
    args = ap.parse_args()
    main(Path(args.input), Path(args.output))
