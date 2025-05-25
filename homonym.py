from flask import Blueprint, request, jsonify
from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer
import torch
import json

homonym_bp = Blueprint('homonym', __name__)
#model = SentenceTransformer("BM-K/KoSimCSE-roberta")
okt = Okt()
model = None

with open("parsed_dictionary.json", "r", encoding="utf-8") as f:
    homonym_embeddings = json.load(f)

def extract_context(tokens, index, window=2):
    start = max(index - window, 0)
    end = min(index + window + 1, len(tokens))
    return ' '.join([word for word, _ in tokens[start:end]])

def get_model():
    global model
    if model is None:
        print("모델 로딩 중...")
        model = SentenceTransformer("BM-K/KoSimCSE-roberta")
        print("모델 로딩 완료")
    return model

def get_best_sense(context_embedding, senses):
    best_sim = -1
    best_sense = None
    for sense in senses:
        examples = sense.get("examples", [])
        if not examples:
            continue
        example_embeddings = get_model().encode(examples, convert_to_tensor=True)
        avg_example_embedding = torch.mean(example_embeddings, dim=0)
        definition_embedding = get_model().encode(sense["definition"], convert_to_tensor=True)
        combined = torch.mean(torch.stack([avg_example_embedding, definition_embedding]), dim=0)
        sim = torch.nn.functional.cosine_similarity(context_embedding, combined, dim=0).item()
        if sim > best_sim:
            best_sim = sim
            best_sense = sense
    return best_sense

def find_all_matches(word):
    return [entry for entry in homonym_embeddings if entry['word'] == word]

def categorize_prompt_text(text):
    tokens = okt.pos(text, stem=True)
    results = []

    for i, (word, tag) in enumerate(tokens):
        semanticCategory = None
        matched_entries = find_all_matches(word)
        if matched_entries:
            if len(matched_entries) > 1:
                context = extract_context(tokens, i)
                context_embedding = get_model().encode(context, convert_to_tensor=True)
                best_entry, best_sense_selected, best_sim = None, None, -1
                for entry in matched_entries:
                    senses = entry.get("senses", [])
                    if not senses:
                        continue
                    best_sense = get_best_sense(context_embedding, senses)
                    if best_sense:
                        example_embeddings = get_model().encode(best_sense.get("examples", []), convert_to_tensor=True)
                        avg_example_embedding = torch.mean(example_embeddings, dim=0)
                        definition_embedding = get_model().encode(best_sense["definition"], convert_to_tensor=True)
                        combined = torch.mean(torch.stack([avg_example_embedding, definition_embedding]), dim=0)
                        sim = torch.nn.functional.cosine_similarity(context_embedding, combined, dim=0).item()
                        if sim > best_sim:
                            best_sim = sim
                            best_entry = entry
                            best_sense_selected = best_sense
                if best_entry:
                    semanticCategory = best_entry.get("semanticCategory")
            else:
                semanticCategory = matched_entries[0].get("semanticCategory")
        results.append({"text": word, "classification": semanticCategory})
    return results

@homonym_bp.route("/homonym", methods=["POST"])
def categorize():
    data = request.get_json()
    prompt = data.get("promptContent", "").strip()
    print(f"[받은 프롬프트]: {prompt}", flush=True)
    if not prompt:
        return jsonify({"error": "No prompt content provided."}), 400
    results = categorize_prompt_text(prompt)
    response_data = {item["text"]: item.get("classification", None) for item in results}
    return jsonify(response_data), 200
