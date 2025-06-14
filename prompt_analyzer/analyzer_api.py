from prompt_analyzer.fluency import compute_fluency
from prompt_analyzer.persistence import compute_persistence
import datetime

DEFAULT_CENTROIDS_PATH = "DPDT/data/centroids.json"

def analyze_from_api(texts, centroids_path=DEFAULT_CENTROIDS_PATH, model_name="skt/kobert-base-v1"):
    print("📥 analyze_from_api 진입", flush=True)

    if not texts:
        print("⚠️ 입력 문장 없음", flush=True)
        return {
            "error": "분석할 문장이 없습니다.",
            "status": 400,
            "timestamp": datetime.datetime.now().isoformat()
        }

    try:
        print("🧠 compute_fluency 시작", flush=True)
        flu_score = compute_fluency(texts, centroids_path, model_name)
        print("✅ compute_fluency 완료:", flu_score, flush=True)

        print("🧠 compute_persistence 시작", flush=True)
        pers_score = compute_persistence(texts, centroids_path, model_name)
        print("✅ compute_persistence 완료:", pers_score, flush=True)

        creativity_score = (flu_score * 0.5 + pers_score * 0.5)
        print("🎯 최종 점수:", creativity_score, flush=True)

        return {
            "fluency": round(flu_score, 4),
            "persistence": round(pers_score, 4),
            "creativity": round(creativity_score, 4),
            "message": "분석이 성공적으로 완료되었습니다.",
            "status": 200,
            "timestamp": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        print("❌ 분석 중 예외 발생:", e, flush=True)
        return {
            "error": str(e),
            "status": 500,
            "timestamp": datetime.datetime.now().isoformat()
        }
