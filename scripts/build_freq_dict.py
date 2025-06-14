# 말뭉치 빈도 사전 생성 스크립트
from prompt_analyzer.preprocessor import clean_text, extract_morphs
from collections import Counter
import json

def build_freq(corpus_path: str, out_path: str):
    freq = Counter()
    total = 0
    with open(corpus_path, "r", encoding="utf8") as f:
        for line in f:
            morphs = extract_morphs(line)
            for w, _ in morphs:
                freq[w] += 1
                total += 1
    rel_freq = {w: count/total for w, count in freq.items()}
    with open(out_path, "w", encoding="utf8") as out:
        json.dump(rel_freq, out, ensure_ascii=False, indent=2)
    print(f"[✔] Saved freq dict → {out_path}")

if __name__ == "__main__":
    for split in ("train", "val", "test"):
        build_freq(f"data/{split}_captions.txt", f"data/{split}_freq.json")
