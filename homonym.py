from flask import Flask, request, jsonify
from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer
import torch
import json

# Flask 앱 생성
app = Flask(__name__)

# 모델 및 형태소 분석기 초기화
model = SentenceTransformer("BM-K/KoSimCSE-roberta")
okt = Okt()

# 동음이의어 및 일반 단어 카테고리 사전 로딩
with open("parsed_dictionary.json", "r", encoding="utf-8") as f:
    homonym_embeddings = json.load(f)

# 문맥 추출 함수
def extract_context(tokens, index, window=2):
    start = max(index - window, 0)
    end = min(index + window + 1, len(tokens))
    return ' '.join([word for word, _ in tokens[start:end]])

# 문맥 임베딩 기반으로 가장 적절한 sense 고르는 함수
def get_best_sense(context_embedding, senses):
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
    return best_sense

# homonym_embeddings 리스트에서 해당 단어 매칭
def find_all_matches(word):
    return [entry for entry in homonym_embeddings if entry['word'] == word]

# 문장을 받아서 단어별 semanticCategory 매칭
def categorize_prompt_text(text):
    tokens = okt.pos(text, stem=True)
    results = []

    for i, (word, tag) in enumerate(tokens):
        semanticCategory = None

        matched_entries = find_all_matches(word)

        if matched_entries:
            if len(matched_entries) > 1:
                context = extract_context(tokens, i)
                context_embedding = model.encode(context, convert_to_tensor=True)

                best_entry = None
                best_sense_selected = None
                best_sim = -1

                for entry in matched_entries:
                    senses = entry.get("senses", [])
                    if not senses:
                        continue
                    best_sense = get_best_sense(context_embedding, senses)
                    if best_sense:
                        example_embeddings = model.encode(best_sense.get("examples", []), convert_to_tensor=True)
                        avg_example_embedding = torch.mean(example_embeddings, dim=0)
                        definition_embedding = model.encode(best_sense["definition"], convert_to_tensor=True)
                        combined = torch.mean(torch.stack([avg_example_embedding, definition_embedding]), dim=0)
                        sim = torch.nn.functional.cosine_similarity(context_embedding, combined, dim=0).item()
                        if sim > best_sim:
                            best_sim = sim
                            best_entry = entry
                            best_sense_selected = best_sense

                if best_entry:
                    semanticCategory = best_entry.get("semanticCategory")

            else:
                # 단일 매칭
                semanticCategory = matched_entries[0].get("semanticCategory")

        results.append({
            "text": word,
            "classification": semanticCategory
        })

    return results

@app.route("/homonym", methods=["POST"])
def categorize():
    data = request.get_json()

    prompt = data.get("promptContent", "").strip()
    print(f"[받은 프롬프트]: {prompt}", flush=True)

    if not prompt:
        return jsonify({"error": "No prompt content provided."}), 400

    results = categorize_prompt_text(prompt)

    # 결과를 단순한 key-value 형태로 가공
    response_data = {}

    for item in results:
        word = item.get("text")
        classification = item.get("classification", None)
        response_data[word] = classification

    return jsonify(response_data), 200
