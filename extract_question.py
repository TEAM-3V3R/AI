import json
import uuid  # ⭐ UUID 생성 모듈 추가

# 1. 원본 JSON 파일 열기
with open("SFTdata.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 2. 필요한 항목만 추출
processed_data = []

for item in raw_data["data_info"]:
    question = item.get("question")
    
    if question:
        processed_data.append({
            "data_id": str(uuid.uuid4()),  # ⭐ 새 UUID 생성
            "question": question
        })

# 3. 결과 저장
with open("processed_data.json", "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

print(f"{len(processed_data)}개의 질문이 저장되었습니다!")
