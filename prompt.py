from flask import Blueprint, request, jsonify

prompt_bp = Blueprint('prompt', __name__)

@prompt_bp.route("/prompt", methods=["POST"])
def echo_prompt():
    data = request.get_json()
    prompt = data.get("promptContent", "").strip()
    print(f"[받은 프롬프트]: {prompt}", flush=True)

    if not prompt:
        return jsonify({"error": "No prompt content provided."}), 400

    return jsonify({
        "echoedPrompt": prompt,
        "message": "프롬프트가 성공적으로 반환되었습니다.",
        "status": 200
    }), 200