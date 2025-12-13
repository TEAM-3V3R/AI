# test/fluency_axes_test.py
"""
Fluency 3-axis test
- Sentence diversity (S)
- Lexical diversity (K)
- Semantic diversity (C)

Purpose:
Verify that S/K/C respond only to intended changes,
and that sentence repetition penalty suppresses false fluency.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from prompt_analyzer.fluency import compute_fluency


CENTROIDS_PATH = "DPDT/models/kmeans_k100/centroids.npy"
MODEL_NAME = "klue/bert-base"


# --------------------------------------------------
# Resource builder
# --------------------------------------------------
def build_resources():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cents = np.load(CENTROIDS_PATH).astype(np.float32)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "centroids": cents,
    }


def run(texts, resources, rep_mode="sentence", rep_strength=1.0, rep_floor=0.15):
    return compute_fluency(
        texts,
        centroids_path=CENTROIDS_PATH,
        model_name=MODEL_NAME,
        resources=resources,
        rep_mode=rep_mode,
        rep_strength=rep_strength,
        rep_floor=rep_floor,
    )


def fmt(label, r):
    return f"[{label:<12}] score={r[0]:6.2f} | S={r[1]:6.2f} K={r[2]:6.2f} C={r[3]:6.2f}"


# --------------------------------------------------
# Test definitions
# --------------------------------------------------
def test_sentence_axis(resources):
    print("\n" + "=" * 72)
    print("AXIS TEST: Sentence diversity (S)")
    print("=" * 72)

    cases = {
        "S-LOW": ["고양이가 있다"],
        "S-MID": ["고양이가 있다", "고양이가 있다"],
        "S-HIGH": ["고양이가 있다"] * 6,
    }

    for k, texts in cases.items():
        print(fmt(k, run(texts, resources)))


def test_lexical_axis(resources):
    print("\n" + "=" * 72)
    print("AXIS TEST: Lexical diversity (K)")
    print("=" * 72)

    cases = {
        "K-LOW": [
            "고양이가 있다",
            "고양이가 있다",
            "고양이가 있다",
        ],
        "K-HIGH": [
            "고양이가 있다",
            "고양이가 조용히 앉아 있다",
            "고양이가 창가에서 잠을 잔다",
        ],
    }

    for k, texts in cases.items():
        print(fmt(k, run(texts, resources)))


def test_semantic_axis(resources):
    print("\n" + "=" * 72)
    print("AXIS TEST: Semantic diversity (C)")
    print("=" * 72)

    cases = {
        "C-LOW": [
            "고양이가 있다",
            "고양이가 앉아 있다",
            "고양이가 잠을 잔다",
        ],
        "C-HIGH": [
            "고양이가 햇볕 아래에서 잠을 잔다",
            "중세 성당의 스테인드글라스가 빛난다",
            "주식 시장의 변동성이 급격히 증가했다",
        ],
    }

    for k, texts in cases.items():
        print(fmt(k, run(texts, resources)))


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    resources = build_resources()
    print("device:", resources["device"])
    print("centroids shape:", resources["centroids"].shape)

    test_sentence_axis(resources)
    test_lexical_axis(resources)
    test_semantic_axis(resources)

    print("\nDone.")


if __name__ == "__main__":
    main()
