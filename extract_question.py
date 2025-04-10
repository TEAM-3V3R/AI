import json

# 원본 데이터 열기
with open("data/SFTdata.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 필요한 항목만 추출
processed_data = []

for idx, item in enumerate(raw_data["data_info"]):
    question = item.get("question")
    
    if question:
        processed_data.append({
            "data_id": idx,  # 리스트 순서대로 번호 부여
            "question": question
        })

# 결과 저장
with open("aiHubData.json", "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

print(f"{len(processed_data)}개의 질문이 저장되었습니다.")
