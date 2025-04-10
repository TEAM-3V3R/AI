from flask import Flask, request, jsonify
from konlpy.tag import Okt
import json

app = Flask(__name__)
okt = Okt()

# homonym.json 파일에서 카테고리 데이터 불러오기
with open("homonym_exaples.json", "r", encoding="utf-8") as f:
    homonym_category_map = json.load(f)

# 형태소 분석 및 카테고리 매칭 함수
def analyze_and_match(text):
    tokens = okt.pos(text, stem=True)
    result = {}

    for word, tag in tokens:
        if word in homonym_category_map:
            result[word] = homonym_category_map[word]

    return result

@app.route("/homonym", methods=["POST"])
def categorize_prompt():
    data = request.get_json()
    prompt = data.get("promptContent", "").strip()
    print(f"[받은 프롬프트]: {prompt}", flush=True)

    if not prompt:
        return jsonify({"error": "No prompt content provided."}), 400

    categorized = analyze_and_match(prompt)

    return jsonify({
        "categorizedWords": categorized,
        "status": 200
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
