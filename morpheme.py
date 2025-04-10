from flask import Flask, request, jsonify
from konlpy.tag import Okt
from collections import Counter

app = Flask(__name__)

# 형태소 분석기 초기화
okt = Okt()

# 형태소 분석 함수
def analyze_pos(text):
    tokens = okt.pos(text, stem=True)  # 형태소 분석 + 어간 추출
    pos_tags = [tag for _, tag in tokens]
    total_count = len(pos_tags)
    counter = Counter(pos_tags)
    ratios = {tag: round(count / total_count, 3) for tag, count in counter.items()}
    return tokens, ratios

@app.route("/morpheme", methods=["POST"])
def analyze_prompt():
    data = request.get_json()

    prompt = data.get("promptContent", "").strip()
    print(f"[받은 프롬프트]: {prompt}", flush=True)

    if not prompt:
        return jsonify({"error": "No prompt content provided."}), 400

    tokens, ratios = analyze_pos(prompt)

    return jsonify({
        "tokens": [{"word": word, "pos": tag} for word, tag in tokens],
        "posRatios": ratios,
        "status": 200
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)