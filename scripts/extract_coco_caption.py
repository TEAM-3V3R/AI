# scripts/extract_coco_caption.py

import json
import sys
from pathlib import Path

# 1) 입력/출력 경로 설정
INPUT  = Path("DPDT/data/MSCOCO_train_val_Korean.json")   # 실제 JSON 파일 위치로 조정하세요
OUTPUT = Path("DPDT/data/coco_caption_ko.txt")

def main():
    # 2) JSON 불러오기
    if not INPUT.is_file():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {INPUT}", file=sys.stderr)
        sys.exit(1)

    with INPUT.open(encoding="utf8") as f:
        raw = json.load(f)

    # 3) annotations 리스트 획득
    if isinstance(raw, dict) and "annotations" in raw:
        anns = raw["annotations"]
    elif isinstance(raw, list):
        anns = raw
    else:
        print(f"❌ 예상치 못한 JSON 구조입니다: {type(raw)}", file=sys.stderr)
        sys.exit(1)

    # 4) 한국어 캡션 추출
    captions = []
    for ann in anns:
        # 실제 파일에서 한국어 캡션 필드명이 'caption_ko' 이므로 그 키를 우선 사용
        if "caption_ko" in ann:
            for sent in ann["caption_ko"]:
                text = sent.strip()
                if text:
                    captions.append(text)
        # 만약 다른 키를 사용한다면 아래 케이스를 참고하세요
        # elif "ko_caption" in ann:
        #     text = ann["ko_caption"].strip()
        #     if text:
        #         captions.append(text)

    # 5) 한 줄에 한 문장씩 출력
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf8") as f:
        for c in captions:
            f.write(c.replace("\n", " ") + "\n")

    print(f"✅ wrote {len(captions)} captions to {OUTPUT}")

if __name__ == "__main__":
    main()
