# prompt_analyzer/analyzer.py
import sys
import json
import argparse
from prompt_analyzer.fluency import compute_fluency


def main():
    parser = argparse.ArgumentParser(
        description="사용자 입력 문장(들)에 대한 Fluency(유창성) 점수를 계산합니다."
    )
    parser.add_argument(
        "-i", "--input",
        nargs="*",
        help="Fluency를 계산할 문장들. 입력하지 않으면 대화형 모드로 전환됩니다."
    )
    parser.add_argument(
        "-c", "--centroids",
        required=True,
        help="학습된 클러스터 센트로이드 파일 경로 (예: data/centroids.npy)"
    )
    parser.add_argument(
        "-m", "--model",
        default="skt/kobert-base-v1",
        help="BERT 모델 이름 (예: skt/kobert-base-v1)"
    )
    args = parser.parse_args()

    # 1) 입력 처리: 인자가 없으면 대화형으로 요청
    if not args.input:
        print("입력하세요 (여러 문장은 엔터로 구분, 완료 시 빈 줄 입력 후 Ctrl+D):")
        texts = []
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    break
                texts.append(line)
        except KeyboardInterrupt:
            pass
    else:
        texts = args.input

    if not texts:
        print("⚠️ 평가할 문장이 입력되지 않았습니다.")
        sys.exit(1)

    # 2) Fluency 계산
    score = compute_fluency(
        texts,
        centroids_path=args.centroids,
        model_name=args.model
    )
    print(f"Fluency 점수: {score:.4f}")


if __name__ == "__main__":
    main()
