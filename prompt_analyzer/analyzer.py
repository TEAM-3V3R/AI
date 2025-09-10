# analyzer.py
from flask import Blueprint, request, jsonify
from prompt_analyzer.analyzer_api import analyze_from_api, DEFAULT_CENTROIDS_PATH

analyzer_bp = Blueprint("analyzer", __name__, url_prefix="/analyzer")

@analyzer_bp.route("/analyze", methods=["POST"])
def analyze_route():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON. Expect an object with 'promptContents'.",
            "status": 400
        }), 400

    texts = data.get("promptContents")
    if not isinstance(texts, list):
        return jsonify({
            "error": "Invalid 'promptContents'. Must be a list of strings.",
            "status": 400
        }), 400
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return jsonify({
            "error": "No non-empty strings in 'promptContents'.",
            "status": 400
        }), 400

    chat_id = data.get("chatId") or data.get("chat_id")
    model_name = data.get("model_name") or "skt/kobert-base-v1"
    centroids_path = data.get("centroids_path") or DEFAULT_CENTROIDS_PATH

    print(f"[Analyzer] n_texts={len(texts)}, model='{model_name}', centroids='{centroids_path}', chat_id={chat_id}", flush=True)

    try:
        result = analyze_from_api(
            texts,
            centroids_path=centroids_path,
            model_name=model_name
        )
    except Exception as e:
        print("analyze_from_api 예외:", str(e), flush=True)
        return jsonify({
            "error": str(e),
            "status": 500
        }), 500


    if chat_id is not None:
        result["chatId"] = chat_id
    result["accepted_key"] = "promptContents"

    code = int(result.get("status", 200))
    return jsonify(result), int(code)
