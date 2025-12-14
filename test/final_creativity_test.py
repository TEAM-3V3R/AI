# test/final_creativity_test.py
"""
Final creativity validation test
- Combines Fluency + Persistence
- Validates 4 quadrants of DPDT space

Quadrants:
1) High Fluency / High Persistence
2) High Fluency / Low Persistence
3) Low Fluency / High Persistence
4) Low Fluency / Low Persistence
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from prompt_analyzer.fluency import compute_fluency
from prompt_analyzer.persistence import compute_persistence


CENTROIDS_PATH = "DPDT/models/kmeans_k100/centroids.npy"
MODEL_NAME = "klue/bert-base"


def build_resources(model_name: str, centroids_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cents = np.load(centroids_path).astype(np.float32)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "centroids": cents,
    }


def run_case(name: str, texts, resources):
    flu, fS, fK, fC = compute_fluency(
        texts,
        centroids_path=CENTROIDS_PATH,
        model_name=MODEL_NAME,
        resources=resources,
    )

    per, pS, pR, pF = compute_persistence(
        texts,
        centroids_path=CENTROIDS_PATH,
        model_name=MODEL_NAME,
        resources=resources,
    )

    creativity = 0.5 * flu + 0.5 * per

    return {
        "name": name,
        "fluency": flu,
        "persistence": per,
        "creativity": creativity,
        "fluency_axes": (fS, fK, fC),
        "persistence_axes": (pS, pR, pF),
    }


def print_case(res):
    print(f"\n[{res['name']}]")
    print(f"  Fluency     : {res['fluency']:.2f}")
    print(f"    └ S/K/C   : {res['fluency_axes'][0]:.2f} / "
          f"{res['fluency_axes'][1]:.2f} / {res['fluency_axes'][2]:.2f}")
    print(f"  Persistence : {res['persistence']:.2f}")
    print(f"    └ S/R/F   : {res['persistence_axes'][0]:.2f} / "
          f"{res['persistence_axes'][1]:.2f} / {res['persistence_axes'][2]:.2f}")
    print(f"  >>> Creativity = {res['creativity']:.2f}")


def main():
    resources = build_resources(MODEL_NAME, CENTROIDS_PATH)
    print("device:", resources["device"])
    print("centroids shape:", resources["centroids"].shape)

    # -------------------------
    # 4 Quadrant Test Cases
    # -------------------------

    # 1) High Fluency / High Persistence
    texts_HF_HP = [
        "안개 낀 숲길을 홀로 걷는 사람",
        "짙은 안개 속에서 조용히 숨을 고르는 인물",
        "차가운 공기와 축축한 흙냄새가 어우러진 숲",
        "숲속의 정적을 따라 천천히 걸어가는 장면",
    ]

    # 2) High Fluency / Low Persistence
    texts_HF_LP = [
        "고양이가 창가에 앉아 있다",
        "주식 시장의 변동성이 커지고 있다",
        "중세 성당의 스테인드글라스가 빛난다",
        "우주 망원경이 새로운 별을 발견했다",
    ]

    # 3) Low Fluency / High Persistence
    texts_LF_HP = [
        "고양이가 있다",
        "고양이가 있다",
        "조용한 고양이가 가만히 있다",
        "작은 고양이가 그 자리에 있다",
    ]

    # 4) Low Fluency / Low Persistence
    texts_LF_LP = [
        "고양이 고양이 고양이",
        "고양이 고양이",
    ]

    cases = [
        ("HF_HP", texts_HF_HP),
        ("HF_LP", texts_HF_LP),
        ("LF_HP", texts_LF_HP),
        ("LF_LP", texts_LF_LP),
    ]

    results = []
    for name, texts in cases:
        results.append(run_case(name, texts, resources))

    print("\n" + "=" * 72)
    print("FINAL CREATIVITY VALIDATION (Fluency + Persistence)")
    print("=" * 72)

    for r in results:
        print_case(r)

    print("\nExpected ordering (rough):")
    print("  HF_HP  >  HF_LP / LF_HP  >  LF_LP")
    print("\nDone.")


if __name__ == "__main__":
    main()
