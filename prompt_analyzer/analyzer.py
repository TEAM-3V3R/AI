from flask import Blueprint, request, jsonify
from prompt_analyzer.analyzer_api import analyze_from_api

analyzer_bp = Blueprint("analyzer", __name__, url_prefix="/analyzer")

@analyzer_bp.route("/analyze", methods=["POST"])
def analyze_route():
    data = request.get_json()
    print("🔥 요청 들어옴:", data, flush=True)

    if not data:
        return jsonify({
            "error": "No input JSON provided.",
            "status": 400
        }), 400

    texts = data.get("texts", [])
    model_name = data.get("model_name", "skt/kobert-base-v1")

    if not texts:
        return jsonify({
            "error": "No 'texts' field provided in request.",
            "status": 400
        }), 400

    try:
        print("🧠 분석 시작", flush=True)
        result = analyze_from_api(texts, model_name=model_name)
        print("✅ 분석 완료:", result, flush=True)

        return jsonify({
            "result": result,
            "message": "분석이 성공적으로 완료되었습니다.",
            "status": 200
        }), 200

    except Exception as e:
        print("❌ 예외 발생:", str(e), flush=True)
        return jsonify({
            "error": str(e),
            "status": 500
        }), 500
