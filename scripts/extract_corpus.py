# scripts/extract_corpus.py

import sys
import argparse
from pathlib import Path
from prompt_analyzer.preprocessor import extract_morphs

def main():
    p = argparse.ArgumentParser(
        description="텍스트 코퍼스에서 형태소 전처리(불용어 제거) 후 토큰 리스트 파일 생성"
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="DPDT/DPDT/data/coco_caption_ko.txt"
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        help="DPDT/DPDT/data/corpus_filtered.txt"
    )
    args = p.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf8") as fin, \
         output_path.open("w", encoding="utf8") as fout:
        num = 0
        for line in fin:
            sent = line.strip()
            if not sent:
                continue
            morphemes = extract_morphs(sent)
            tokens = [w for w, _ in morphemes]
            if tokens:
                fout.write(" ".join(tokens) + "\n")
                num += 1

    print(f"✅ 전처리 {num}문장 → {output_path}")

if __name__ == "__main__":
    main()
