# 의미 토큰 추출 모듈 : extract_meaning_tokens 함수 구현 예정

from prompt_analyzer.preprocessor import extract_morphs

def extract_meaning_tokens(text: str) -> list[str]:
    """
    주어진 텍스트에서 형태소 분석 → 불용어 필터링 → 
    명사(NN*), 동사(VV*), 형용사(VA*) 어근만 추출하여 리스트로 반환.
    """
    morphs = extract_morphs(text)
    # NN*, VV*, VA* 어근만
    tokens = [w for w, pos in morphs
              if pos.startswith(("NN", "VV", "VA"))]
    return tokens