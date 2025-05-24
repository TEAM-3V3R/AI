from flask import Blueprint, request, jsonify
from konlpy.tag import Okt
from collections import Counter

morpheme_bp = Blueprint('morpheme', __name__)
okt = Okt()

def analyze_pos(text):
    tokens = okt.pos(text, stem=True)
    pos_tags = [tag for _, tag in tokens]
    total_count = len(pos_tags)
    counter = Counter(pos_tags)
    ratios = {tag: round(count / total_count, 3) for tag, count in counter.items()}
    return tokens, ratios

@morpheme_bp.route("/morpheme", methods=["POST"])
def analyze_prompt():
    data = request.get_json()
    prompt = data.get("promptContent", "").strip()
    print(f"[받은 프롬프트]: {prompt}", flush=True)

    if not prompt:
        return jsonify({"error": "No prompt content provided."}), 400

    tokens, ratios = analyze_pos(prompt)
    response_data = {
        "josaSum": ratios.get("Josa", 0.0),
        "nounSum": ratios.get("Noun", 0.0),
        "verbSum": ratios.get("Verb", 0.0)
    }
    return jsonify(response_data), 200
