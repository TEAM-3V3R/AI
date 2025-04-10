import json
from konlpy.tag import Okt

okt = Okt()

# 데이터 불러오기
with open('data/aiHubData.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 분석 결과 저장용 리스트
results = []

for item in data:
    question = item['question']
    data_id = item['data_id']
    pos_tags = okt.pos(question) # 전체 품사 태깅

    results.append({
        "data_id": data_id,
        "question": question,
        "pos_tags": pos_tags # 필터 없이 전체 저장
    })

# 결과를 JSON 파일로 저장
with open('data/morpheme_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("형태소 분석 결과가 'morpheme_results.json'에 저장되었습니다.")
