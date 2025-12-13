# test/analyzer_api_test.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prompt_analyzer.analyzer_api import analyze_from_api

texts = [
    "안개가 자욱한 숲길에서 한 사람이 새벽의 고요한 공기 속을 홀로 걷고 있다.",
    "밝은 햇빛이 내리쬐는 푸른 들판에서 강아지가 자유롭게 달리며 뛰어논다.",
    "네온 불빛으로 가득한 도시의 밤거리에서 자동차가 빠른 속도로 질주하고 있다.",
    "늦은 오후의 공원에서 아이들이 웃으며 서로를 쫓아다니며 뛰어놀고 있다.",
]

result = analyze_from_api(texts)

print("\n=== ANALYZE RESULT ===")
for k, v in result.items():
    print(k, ":", v)
