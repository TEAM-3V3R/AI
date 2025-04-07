from konlpy.tag import Okt
from collections import Counter

def analyze_pos(text):
    okt = Okt()
    tokens = okt.pos(text, stem=True)

    pos_tags = [tag for _, tag in tokens]
    total_count = len(pos_tags)

    counter = Counter(pos_tags)
    ratios = {tag : round(count / total_count, 3) for tag, count in counter.items()}

    return tokens, ratios

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