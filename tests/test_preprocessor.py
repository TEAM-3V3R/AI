# tests/test_preprocessor.py

import re
import pytest
from prompt_analyzer.preprocessor import clean_text, extract_morphs, WORD_STOP, STOP_POS

@pytest.mark.parametrize("input_txt, expected", [
    # clean_text 테스트: 특수문자 제거, 띄어쓰기 정리
    ("안개낀숲길@@위에!! 달빛", "안개 낀 숲길 위에 달빛"),
    ("  여러   공백\t테스트  ", "여러 공백 테스트"),
])
def test_clean_text(input_txt, expected):
    assert clean_text(input_txt) == expected

def test_word_stop_contains_and_pos():
    # WORD_STOP 에 최소한 "그리고" 가 있어야 불용어로 걸러짐
    assert "그리고" in WORD_STOP
    # STOP_POS 에 조사 JKS 가 포함되어야 함
    assert "JKS" in STOP_POS

def test_extract_morphs_filters():
    text = "안개 낀 숲길 위에 그리고 은은한 달빛이 비친다."
    morphs = extract_morphs(text)
    # “그리고” 와 조사 “에” 는 필터링되어야 함
    words = [w for w, _ in morphs]
    assert "그리고" not in words
    assert "에" not in words
    # 남은 토큰 중에는 “안개”, “숲길”, “달빛” 이 있어야 함
    assert "안개" in words
    assert "숲길" in words
    assert "달빛" in words

def test_extract_morphs_empty():
    # 빈 문자열이나 의미 없으면 빈 리스트
    assert extract_morphs("") == []
    assert extract_morphs("이다의") == []  # “이다”만 어미라 필터링

