# prompt_analyzer/analyzer.py

import os
import sys
import json
import argparse

from prompt_analyzer.fluency import compute_fluency
from prompt_analyzer.persistence import compute_persistence

def main():
    parser = argparse.ArgumentParser(
        description="입력 문장(들)에 대한 Fluency(유창성)과 Persistence(지속성) 점수를 계산합니다."
    )
    parser.add_argument(
        "-i", "--input",
        nargs="*",
        help="Fluency/Persistence를 계산할 문장들. 입력하지 않으면 대화형 모드로 전환됩니다."
    )
    parser.add_argument(
        "-c", "--centroids",
        required=True,
        help="학습된 클러스터 센트로이드 파일 경로 (예: data/centroids.npy)."
    )
    parser.add_argument(
        "-m", "--model",
        default="skt/kobert-base-v1",
        help="BERT 모델 이름 (예: skt/kobert-base-v1)."
    )
    args = parser.parse_args()

    # (1) 입력 텍스트를 수집 (대화형 또는 인자로부터)
    if not args.input:
        print(">>> 평가할 문장들을 입력하세요. (여러 문장은 엔터로 구분, 빈 줄 후 Ctrl+D ↓)")
        texts = []
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:       # 빈 줄 입력 시 입력 종료
                    break
                texts.append(line)
        except KeyboardInterrupt:
            pass
    else:
        texts = args.input

    if not texts:
        print("⚠️ 평가할 문장이 입력되지 않았습니다.")
        sys.exit(1)

    # (2) Fluency 계산
    try:
        flu_score = compute_fluency(
            texts,
            centroids_path=args.centroids,   # ← 반드시 centroid 파일 경로를 넘겨 줘야 합니다.
            model_name=args.model
            # 만약 tokens.txt 를 미리 만들어 두었다면 tokens_path="data/tokens.txt" 추가 가능
        )
        print(f"• Fluency(유창성) 점수: {flu_score:.4f}")
    except Exception as e:
        print(f"❌ Fluency 계산 중 오류가 발생했습니다: {e}")
        sys.exit(1)

    # (3) Persistence 계산
    try:
        pers_score = compute_persistence(
            texts_or_idlists=texts,
            centroids_path=args.centroids,   # 동일한 centroid 파일을 사용
            model_name=args.model
        )
        print(f"• Persistence(지속성) 점수: {pers_score:.4f}")
    except Exception as e:
        print(f"❌ Persistence 계산 중 오류가 발생했습니다: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
