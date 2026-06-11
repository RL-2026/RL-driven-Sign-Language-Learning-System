"""
make_dpo_dataset.py
chosen 포맷: 자연어 설명 + GLOSSES: 태그 라인
rejected 포맷: 잘못된 분해 3가지 유형
"""

import json
import os
import random

# 글로스별 간단 의미 설명 생성기 (사전에 없으면 단어 자체를 그대로 사용)
GLOSS_DESCRIPTIONS = {
    "교통": "차량이나 사람이 도로를 이동하는 행위",
    "사고": "예상치 못하게 발생한 나쁜 사건",
    "학교": "학생들이 공부하는 교육 기관",
    "생활": "일상적으로 살아가는 방식",
    "남자": "성별이 남성인 사람",
    "여자": "성별이 여성인 사람",
    "학생": "학교에서 공부하는 사람",
    "가족": "혈연이나 결혼으로 맺어진 공동체",
    "회의": "여러 사람이 모여 의견을 나누는 자리",
    "의사": "병을 진단하고 치료하는 전문 직업",
    "치과": "이와 잇몸을 치료하는 전문 의료 기관",
}

def _get_desc(gloss: str) -> str:
    """글로스에 대한 짧은 설명 반환. 없으면 기본 문구 생성."""
    return GLOSS_DESCRIPTIONS.get(gloss, f"{gloss}의 의미를 가진 수어 단어")


def _build_chosen(target_word: str, parts: list) -> str:
    """
    설명형 chosen 포맷 생성.
    마지막 줄은 반드시 'GLOSSES: 단어1 단어2' 형태.
    """
    # 조사 처리 (은/는)
    last_char = target_word[-1]
    josa = "은" if (ord(last_char) - 0xAC00) % 28 != 0 else "는"

    # 분해 표현 (A와 B로 / A, B, C로)
    if len(parts) == 2:
        decomp_expr = f"[{parts[0]}]과 [{parts[1]}]"
    else:
        decomp_expr = ", ".join(f"[{p}]" for p in parts)

    lines = [f"{target_word}{josa} {decomp_expr}로 분해됩니다."]
    for p in parts:
        lines.append(f"{p}: {_get_desc(p)}")
    lines.append(f"GLOSSES: {' '.join(parts)}")

    return "\n".join(lines)


def _build_rejected_variants(target_word: str, chosen_parts: list, allowed_glosses: set) -> list:
    """
    rejected 후보 3가지 생성.
    1) 글자 단위 분해
    2) 미등록 단어 hallucination 시뮬레이션
    3) 잘못된 분리 위치
    """
    rejected = []

    # 유형 1: 글자 단위 분해 (항상 생성)
    char_parts = list(target_word)
    r1_lines = [
        f"{target_word}는 각 글자로 분해됩니다.",
    ]
    for c in char_parts:
        r1_lines.append(f"{c}: 개별 글자")
    r1_lines.append(f"GLOSSES: {' '.join(char_parts)}")
    rejected.append("\n".join(r1_lines))

    # 유형 2: hallucination — 사전에 없는 단어로 분해
    fake_word = target_word + "하기"
    if fake_word not in allowed_glosses:
        r2_lines = [
            f"{target_word}는 [{fake_word}]로 분해됩니다.",
            f"{fake_word}: {target_word}와 관련된 행위",
            f"GLOSSES: {fake_word}",
        ]
        rejected.append("\n".join(r2_lines))

    # 유형 3: 잘못된 분리 위치 (1글자 + 나머지)
    if len(target_word) >= 3:
        wrong_left  = target_word[:1]
        wrong_right = target_word[1:]
        wrong_parts = [wrong_left, wrong_right]
        if wrong_parts != chosen_parts:
            r3_lines = [
                f"{target_word}는 [{wrong_left}]과 [{wrong_right}]로 분해됩니다.",
                f"{wrong_left}: 첫 글자",
                f"{wrong_right}: 나머지 부분",
                f"GLOSSES: {' '.join(wrong_parts)}",
            ]
            rejected.append("\n".join(r3_lines))

    return rejected


class DPODatasetGenerator:
    def __init__(self, dict_path="../sign_dict.txt"):
        self.dict_path = dict_path
        self.allowed_glosses = set()
        self.sorted_glosses  = []
        self._load_dictionary()

    def _load_dictionary(self):
        if not os.path.exists(self.dict_path):
            raise FileNotFoundError(f"❌ {self.dict_path} 파일을 찾을 수 없습니다.")

        with open(self.dict_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    self.allowed_glosses.add(word)

        self.sorted_glosses = sorted(
            list(self.allowed_glosses), key=len, reverse=True
        )
        print(f"📚 사전 로드 완료: 총 {len(self.allowed_glosses):,}개 단어")

    def find_combinations(self):
        dpo_pairs = []
        noise_keywords = [
            "지부", "센터", "협회", "통역", "지원", "청",
            "도청", "시청", "구청", "대학교", "시장", "특별자치시",
        ]
        print("🔍 설명형 DPO 골드셋 생성 중...")

        for target_word in self.sorted_glosses:
            if len(target_word) < 2:
                continue
            if any(n in target_word for n in noise_keywords):
                continue

            matched = False

            # 1단계: 2분할 완전 일치 매칭 (교통사고 → 교통 + 사고)
            for i in range(1, len(target_word)):
                left  = target_word[:i]
                right = target_word[i:]
                if len(left) >= 2 and len(right) >= 2:
                    if left in self.allowed_glosses and right in self.allowed_glosses:
                        if not any(n in left or n in right for n in noise_keywords):
                            self._add_pair(dpo_pairs, target_word, [left, right])
                            matched = True
                            break

            if matched:
                continue

            # 2단계: 남/여 축약어 규칙 (남학생 → 남자 + 학생)
            if len(target_word) >= 2 and target_word[0] in ['남', '여']:
                prefix_char   = target_word[0]
                suffix_text   = target_word[1:]
                expanded_prefix = "남자" if prefix_char == '남' else "여자"
                if (
                    suffix_text in self.allowed_glosses
                    and len(suffix_text) >= 2
                    and expanded_prefix in self.allowed_glosses
                ):
                    self._add_pair(dpo_pairs, target_word, [expanded_prefix, suffix_text])

        return dpo_pairs

    def _add_pair(self, dpo_pairs, target_word, chosen_parts):
        prompt = (
            f"다음 복합어를 수어 사전에 등록된 단어 단위로 분해하세요.\n"
            f"입력 단어: {target_word}"
        )

        # ── chosen: 설명형 포맷 ──
        chosen = _build_chosen(target_word, chosen_parts)

        # ── rejected: 최대 2개 ──
        rejected_variants = _build_rejected_variants(
            target_word, chosen_parts, self.allowed_glosses
        )

        for rejected in rejected_variants[:2]:
            dpo_pairs.append({
                "prompt":   prompt,
                "chosen":   chosen,
                "rejected": rejected,
            })

    def save_dataset(self, output_path="sign_dpo_dataset.json"):
        pairs = self.find_combinations()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 설명형 DPO 데이터셋 저장 완료: {output_path}")
        print(f"📊 총 {len(pairs)}개 학습 쌍 생성")

        # 샘플 출력
        print("\n── 샘플 확인 (첫 번째 쌍) ──")
        if pairs:
            print("[PROMPT]")
            print(pairs[0]["prompt"])
            print("\n[CHOSEN]")
            print(pairs[0]["chosen"])
            print("\n[REJECTED]")
            print(pairs[0]["rejected"])


if __name__ == "__main__":
    generator = DPODatasetGenerator()
    generator.save_dataset()