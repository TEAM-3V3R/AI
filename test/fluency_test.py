# test/fluency_test.py
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


def run_case(texts, resources, rep_mode="sentence", rep_strength=1.0, rep_floor=0.15):
    # rep_*는 fluency.py 최종 설계에 맞춰 sentence만 사용
    score, S, K, C = compute_fluency(
        texts,
        centroids_path=CENTROIDS_PATH,
        model_name=MODEL_NAME,
        resources=resources,
        floor_k=0.20,
        rep_mode=rep_mode,
        rep_strength=float(rep_strength),
        rep_floor=float(rep_floor),
    )
    return score, S, K, C


def fmt_row(label, x):
    return f"[{label:<10}] score={x[0]:6.2f} | S={x[1]:6.2f} K={x[2]:6.2f} C={x[3]:6.2f}"


def main():
    resources = build_resources(MODEL_NAME, CENTROIDS_PATH)
    print("device:", resources["device"])
    print("centroids shape:", resources["centroids"].shape)

    # 케이스 1: 매우 단조 (문장 적고, 단어 반복)
    texts_low = [
        "고양이 고양이 고양이",
        "고양이 고양이",
    ]

    # 케이스 2: 문장/어휘/의미가 더 다양
    texts_high = [
        "안개 낀 숲길을 홀로 걷는 사람",
        "강아지가 뛰노는 푸른 들판",
        "도시의 밤거리를 달리는 자동차",
        "아이들이 공원에서 뛰어노는 장면",
    ]

    # 케이스 3: 같은 문장 반복 (DPDT 관점에서 유연성 과대평가 방지 타겟)
    texts_many_sent = [
        "고양이가 있다",
        "고양이가 있다",
        "고양이가 있다",
        "고양이가 있다",
        "고양이가 있다",
        "고양이가 있다",
        "고양이가 있다",
        "고양이가 있다",
    ]

    cases = [
        ("LOW", texts_low),
        ("HIGH", texts_high),
        ("MANY_SENT", texts_many_sent),
    ]

    # -----------------------------------------
    # A) Baseline: 패널티 없음 (rep_mode="none")
    # -----------------------------------------
    print("\n" + "=" * 72)
    print("BASELINE (no repetition penalty): rep_mode='none'")
    print("=" * 72)

    base = {}
    for name, texts in cases:
        out = run_case(texts, resources, rep_mode="none", rep_strength=0.0, rep_floor=0.15)
        base[name] = out
        print(fmt_row(name, out))

    # -----------------------------------------
    # B) Final(default): sentence repetition penalty
    # -----------------------------------------
    FINAL_MODE = "sentence"
    FINAL_STRENGTH = 1.0
    FINAL_FLOOR = 0.15

    print("\n" + "=" * 72)
    print(f"FINAL (sentence repetition penalty): rep_mode='{FINAL_MODE}', "
          f"rep_strength={FINAL_STRENGTH}, rep_floor={FINAL_FLOOR}")
    print("=" * 72)

    final = {}
    for name, texts in cases:
        out = run_case(texts, resources, rep_mode=FINAL_MODE, rep_strength=FINAL_STRENGTH, rep_floor=FINAL_FLOOR)
        final[name] = out
        print(fmt_row(name, out))

    print("\nΔ(score) FINAL - BASELINE:")
    for name in ["LOW", "HIGH", "MANY_SENT"]:
        d = final[name][0] - base[name][0]
        print(f"  {name:<10}: {d:+.2f}")

    # -----------------------------------------
    # C) Sensitivity sweep (optional)
    #    sentence 패널티 강도만 바꿔보기
    # -----------------------------------------
    strengths = [0.5, 1.0, 1.5, 2.0]
    print("\n" + "=" * 72)
    print("SENSITIVITY (sentence penalty strength sweep)")
    print("=" * 72)

    for s in strengths:
        print(f"\n-- rep_strength={s} (rep_mode='sentence', rep_floor={FINAL_FLOOR})")
        for name, texts in cases:
            out = run_case(texts, resources, rep_mode="sentence", rep_strength=s, rep_floor=FINAL_FLOOR)
            print(fmt_row(name, out))

        # 방향성 체크(필수 조건 느낌으로)
        if not (run_case(texts_high, resources, "sentence", s, FINAL_FLOOR)[0]
                > run_case(texts_low, resources, "sentence", s, FINAL_FLOOR)[0]):
            print("⚠️  WARNING: HIGH score not > LOW score (unexpected)")

    print("\nDone.")


if __name__ == "__main__":
    main()
