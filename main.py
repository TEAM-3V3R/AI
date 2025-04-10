from flask import Flask, request, jsonify
from konlpy.tag import Okt
from collections import Counter
from sentence_transformers import SentenceTransformer
import torch
import json

app = Flask(__name__)

# 모델 및 형태소 분석기 초기화
model = SentenceTransformer("BM-K/KoSimCSE-roberta")
okt = Okt()

# 동음이의어 임베딩 불러오기
with open("data/homonym_embeddings.json", "r", encoding="utf-8") as f:
    homonym_embeddings = json.load(f)

# 형태소 분석
def analyze_pos(text):
    # okt = Okt()
    tokens = okt.pos(text, stem=True)
    pos_tags = [tag for _, tag in tokens]
    total_count = len(pos_tags)
    counter = Counter(pos_tags)
    ratios = {tag: round(count / total_count, 3) for tag, count in counter.items()}
    return tokens, ratios

# 문맥 추출
def extract_context(tokens, index, window=2):
    start = max(index - window, 0)
    end = min(index + window + 1, len(tokens))
    return ' '.join([word for word, _ in tokens[start:end]])

# 동음이의어 의미 추론
def get_best_sense_by_example_avg(context_embedding, senses):
    best_sim = -1
    best_sense = None
    for sense in senses:
        examples = sense.get("examples", [])
        if not examples:
            continue
        example_embeddings = model.encode(examples, convert_to_tensor=True)
        avg_example_embedding = torch.mean(example_embeddings, dim=0)
        definition_embedding = model.encode(sense["definition"], convert_to_tensor=True)
        combined = torch.mean(torch.stack([avg_example_embedding, definition_embedding]), dim=0)
        sim = torch.nn.functional.cosine_similarity(context_embedding, combined, dim=0).item()
        if sim > best_sim:
            best_sim = sim
            best_sense = sense
    return best_sense, best_sim

# 스프링에서 받는 POST 요청 처리
@app.route("/process", methods=["POST"])
def process_prompt():
    data = request.get_json()
    
    # Spring에서는 "promptContent"로 보냄
    prompt = data.get("promptContent", "").strip()

    print(f"[📥 받은 프롬프트]: {prompt}", flush=True)

    if not prompt:
        return jsonify({"error": "No prompt content provided."}), 400

    # 형태소 분석 및 의미 추론 (실제로 결과 반환은 안함)
    tokens, pos_ratios = analyze_pos(prompt)

    for i, (word, tag) in enumerate(tokens):
        if word not in homonym_embeddings:
            continue

        context = extract_context(tokens, i)
        context_embedding = model.encode(context, convert_to_tensor=True)
        senses = homonym_embeddings[word]
        best_sense, _ = get_best_sense_by_example_avg(context_embedding, senses)

        # 실제 응답에 포함하진 않지만 내부 처리를 수행함

    # 성공 응답만 반환
    return jsonify({
        "message": "프롬프트가 성공적으로 처리되었습니다.",
        "status": 200
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
