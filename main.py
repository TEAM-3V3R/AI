from konlpy.tag import Okt
from collections import Counter
from sentence_transformers import SentenceTransformer
import torch
import json
from tqdm import tqdm

# 문장 임베딩 모델 로딩 (KoSimCSE)
model = SentenceTransformer("BM-K/KoSimCSE-roberta")
okt = Okt()

# 동음이의어 예문 + 임베딩 캐싱 불러오기
with open("data/homonym_embeddings.json", "r", encoding="utf-8") as f:
    homonym_embeddings = json.load(f)

def analyze_pos(text):
    # okt = Okt()
    tokens = okt.pos(text, stem=True)

    pos_tags = [tag for _, tag in tokens]
    total_count = len(pos_tags)

    counter = Counter(pos_tags)
    ratios = {tag : round(count / total_count, 3) for tag, count in counter.items()}

    return tokens, ratios

def extract_context(tokens, index, window=2):
    """지정된 단어 인덱스 기준 앞뒤 window 수만큼 문맥을 문자열로 리턴"""
    start = max(index - window, 0)
    end = min(index + window + 1, len(tokens))
    return ' '.join([word for word, _ in tokens[start:end]])

def get_best_sense_by_example_avg(context_embedding, senses):
    """각 의미의 예문 전체 평균 벡터와 context를 비교"""
    best_sim = -1
    best_sense = None

    for sense in senses:
        examples = sense.get("examples", [])
        if not examples:
            continue

        example_embeddings = model.encode(examples, convert_to_tensor=True)
        avg_example_embedding = torch.mean(example_embeddings, dim=0)
        definition_embedding = model.encode(sense["definition"], convert_to_tensor=True)

        # 정의와 예문 평균 결합
        combined = torch.mean(torch.stack([avg_example_embedding, definition_embedding]), dim=0)

        sim = torch.nn.functional.cosine_similarity(context_embedding, combined, dim=0).item()

        if sim > best_sim:
            best_sim = sim
            best_sense = sense

    return best_sense, best_sim

if __name__ == "__main__":
    while True:
        text = input("프롬프트 입력(끝내려면 exit 입력) : ").strip()

        if text.lower() in ['exit']:
            print("끝")
            break

        tokens, pos_ratios = analyze_pos(text)

        print("\n[프롬프트]")
        print(text)

        print("\n[형태소 + 품사]")
        for word, tag in tokens:
            print(f"{word} : {tag}")

        print("\n[품사 비율]")
        # print(pos_ratios)
        for tag, ratio in pos_ratios.items():
            print(f"{tag} : {ratio}")

        # 동음이의어 의미 추론
        #candidate_words = [word for word, tag in tokens if word in homonym_embeddings]
#
        #if candidate_words:
        #    print("\n[동음이의어 의미 추론 결과]")
        #    prompt_embedding = model.encode(text, convert_to_tensor=True)
#
        #    for word in candidate_words:
        #        senses = homonym_embeddings[word]
        #        similarities = []
#
        #        for sense in senses:
        #            if "embedding" not in sense:
        #                continue  # 예외 처리
#
        #            sense_embedding = torch.tensor(sense["embedding"])
        #            sim = torch.nn.functional.cosine_similarity(prompt_embedding, sense_embedding, dim=0)
        #            similarities.append((sim.item(), sense))
        #        
        #        if similarities:
        #            best_match = max(similarities, key=lambda x: x[0])[1]
#
        #            print(f"\n 단어: '{word}'")
        #            print(f"→ 의미 정의: {best_match['definition']}")
        #            print(f"→ 카테고리: {best_match['semanticCategory']}")
        #        else:
        #            print("\n단어 '{word}'의 임베딩 데이터가 없습니다.")
        #else:
        #    print("\n[동음이의어 없음] 형태소 분석 결과에 등록된 동음이의어가 없습니다.")
        # 의미 추론
        print("\n[동음이의어 의미 추론 결과]")
        for i, (word, tag) in enumerate(tokens):
            if word not in homonym_embeddings:
                continue

            context = extract_context(tokens, i, window=2)
            context_embedding = model.encode(context, convert_to_tensor=True)

            senses = homonym_embeddings[word]
            best_sense, sim = get_best_sense_by_example_avg(context_embedding, senses)

            if best_sense:
                print(f"\n 단어 위치 {i}: '{word}' (문맥: \"{context}\")")
                print(f"→ 의미 정의: {best_sense['definition']}")
                print(f"→ 카테고리: {best_sense['semanticCategory']}")
                print(f"→ 유사도 점수: {sim:.3f}")
            else:
                print(f"\n 단어 위치 {i}: '{word}' → 적절한 의미를 찾을 수 없습니다.")