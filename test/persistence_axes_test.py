# test/persistence_axes_test.py
"""
Persistence 3-axis test
- S: modifier-based elaboration (심화/수식어 풍부도)
- R: lexical concentration (어휘 집중도, Simpson)
- F: semantic focus (의미/클러스터 집중도, Simpson)

Goal:
One axis changes while others are kept as stable as possible.
This is a sanity & directionality test, not a strict unit test.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

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


def run_case(name: str, texts, resources, **kwargs):
    score, S, R, F = compute_persistence(
        texts,
        centroids_path=CENTROIDS_PATH,
        model_name=MODEL_NAME,
        resources=resources,
        **kwargs,
    )
    return score, S, R, F


def fmt(label: str, res):
    return f"[{label:<12}] score={res[0]:6.2f} | S={res[1]:6.2f} R={res[2]:6.2f} F={res[3]:6.2f}"


# -------------------------
# AXIS TESTS
# -------------------------
def axis_test_S(resources):
    """
    S axis: Increase modifiers (형용사/부사/관형사) while keeping topic similar.
    Expect: S goes up, R/F not necessarily huge change, score tends to go up.
    """
    print("\n" + "=" * 72)
    print("AXIS TEST: S (modifier-based elaboration)")
    print("=" * 72)

    S_LOW = [
        "고양이가 있다",
        "고양이가 있다",
        "고양이가 있다",
    ]
    S_HIGH = [
        "작고 귀여운 고양이가 조용히 있다",
        "매우 부드럽고 하얀 고양이가 창가에 가만히 있다",
        "조용히 숨을 고르는 고양이가 따뜻한 햇빛 아래에 있다",
    ]

    low = run_case("S-LOW", S_LOW, resources)
    high = run_case("S-HIGH", S_HIGH, resources)

    print(fmt("S-LOW", low))
    print(fmt("S-HIGH", high))

    if not (high[1] > low[1]):
        print("⚠️  WARNING: S did not increase as expected. Check POS tags / extractor.")


def axis_test_R(resources):
    """
    R axis: Lexical concentration.
    Make one case highly repetitive (same token repeated),
    and another case more varied but same rough topic.
    Expect: R is higher in repetitive case.
    """
    print("\n" + "=" * 72)
    print("AXIS TEST: R (lexical concentration)")
    print("=" * 72)

    R_HIGH = [
        "고양이 고양이 고양이 고양이",
        "고양이 고양이 고양이",
        "고양이 고양이",
    ]
    R_LOW = [
        "고양이가 창가에 앉아 있다",
        "고양이가 천천히 걸어간다",
        "고양이가 조용히 잠든다",
    ]

    high = run_case("R-HIGH", R_HIGH, resources)
    low = run_case("R-LOW", R_LOW, resources)

    print(fmt("R-HIGH", high))
    print(fmt("R-LOW", low))

    if not (high[2] > low[2]):
        print("⚠️  WARNING: R did not increase as expected. Check tokenization / tau_r / r_floor.")


def axis_test_F(resources):
    """
    F axis: Semantic focus (cluster concentration).
    F-HIGH: stay in one semantic topic (all about cats)
    F-LOW : spread across different topics (cats, finance, cathedral, science)
    Expect: F is higher in focused case, lower in spread case.
    """
    print("\n" + "=" * 72)
    print("AXIS TEST: F (semantic cluster focus)")
    print("=" * 72)

    F_HIGH = [
        "고양이가 햇빛 아래에서 잠든다",
        "고양이가 창가에서 몸을 말고 쉰다",
        "고양이가 조용히 숨을 고른다",
        "고양이가 담요 위에 웅크린다",
        "고양이가 따뜻한 바닥에 누워 있다",
    ]
    F_LOW = [
        "고양이가 햇빛 아래에서 잠든다",
        "주식 시장의 변동성이 확대되었다",
        "중세 성당의 스테인드글라스가 빛난다",
        "양자 물리학의 확률 해석",
        "고대 신화 속 창조 이야기",
    ]

    high = run_case("F-HIGH", F_HIGH, resources)
    low = run_case("F-LOW", F_LOW, resources)

    print(fmt("F-HIGH", high))
    print(fmt("F-LOW", low))

    if not (high[3] > low[3]):
        print("⚠️  WARNING: F did not increase as expected. "
              "This can happen with very small N or weak centroid separation; try longer texts.")


def main():
    resources = build_resources(MODEL_NAME, CENTROIDS_PATH)
    print("device:", resources["device"])
    print("centroids shape:", resources["centroids"].shape)

    axis_test_S(resources)
    axis_test_R(resources)
    axis_test_F(resources)

    print("\nDone.")


if __name__ == "__main__":
    main()
