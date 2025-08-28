import argparse, json, re
from pathlib import Path

def clean_text(s: str) -> str:
    """공백 정리 및 단순 클린업"""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def main(root: Path, out: Path, append: bool = False):
    out.parent.mkdir(parents=True, exist_ok=True)
    captions, n_json, n_err = set(), 0, 0

    # append 모드라면 기존 결과를 먼저 읽어서 set에 추가
    if append and out.exists():
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                line = clean_text(line)
                if line:
                    captions.add(line)

    # JSON 순회
    for fp in root.rglob("*.json"):
        n_json += 1
        try:
            txt = fp.read_text(encoding="utf-8")
            data = json.loads(txt)
        except Exception:
            try:
                txt = fp.read_text(encoding="utf-8-sig", errors="ignore")
                data = json.loads(txt)
            except Exception:
                n_err += 1
                continue

        # caption 필드 추출
        if "caption" in data:
            c = clean_text(data["caption"])
            if c:
                captions.add(c)

    # 루프 끝난 후 한 번만 저장
    with out.open("w", encoding="utf-8") as f:
        for c in sorted(captions):
            f.write(c + "\n")

    print(f"[done] scanned={n_json} json, errors={n_err}")
    print(f"       captions={len(captions)} -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="JSON들이 들어있는 루트 폴더")
    ap.add_argument("--out", type=Path, default=Path("data/sources/captions_aihub_webtoon.txt"))
    ap.add_argument("--append", action="store_true", help="기존 파일에 이어쓰기")
    args = ap.parse_args()

    main(args.root, args.out, append=args.append)