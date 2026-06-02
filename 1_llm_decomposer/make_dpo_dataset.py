import json
import os

class DPODatasetGenerator:
    def __init__(self, dict_path="../sign_dict.txt"):
        self.dict_path = dict_path
        self.allowed_glosses = set()
        self.sorted_glosses = []
        self._load_dictionary()

    def _load_dictionary(self):
        """15,000개 사전을 예외 처리와 함께 안전하게 로드합니다."""
        if not os.path.exists(self.dict_path):
            raise FileNotFoundError(f"❌ {self.dict_path} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
            
        with open(self.dict_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:  # 빈 줄이 아니라면 사전에 추가
                    self.allowed_glosses.add(word)
                    
        # 단어 길이가 긴 순서대로 정렬 (글자 수가 많은 복합어부터 탐색하기 위함)
        self.sorted_glosses = sorted(list(self.allowed_glosses), key=len, reverse=True)
        print(f"📚 사전 로드 완료: 총 {len(self.allowed_glosses):,}개의 단어가 준비되었습니다.")

    def find_combinations(self):
        """사전 내 단어들을 조합하여 복합어 패턴을 분석합니다."""
        dpo_pairs = []
        print("🔍 사전 내부에서 복합어 패턴 분석 중...")

        for target_word in self.sorted_glosses:
            # 최소 3글자 이상의 복합어 타겟팅 (예: 가족회의, 치과의사 등)
            if len(target_word) < 3:
                continue

            # 단어를 두 조각으로 쪼개어 양쪽 모두 사전에 존재하는지 확인
            for i in range(1, len(target_word)):
                left = target_word[:i]
                right = target_word[i:]

                # 쪼갠 두 단어가 모두 15,000개 사전에 존재하는 황금 조합인 경우
                if left in self.allowed_glosses and right in self.allowed_glosses:
                    prompt = f"다음 복합어를 수어 사전에 등록된 단어 단위로 분해하세요.\n입력 단어: {target_word}"
                    
                    # 정답 (Chosen): 사전에 존재하는 의미 단위 조합 (예: ["가족", "회의"])
                    chosen = json.dumps([left, right], ensure_ascii=False)
                    
                    # 오답 (Rejected): 의미를 무시하고 낱글자로 찢어버린 형태 (예: ["가", "족", "회", "의"])
                    char_split = [char for char in target_word]
                    rejected = json.dumps(char_split, ensure_ascii=False)

                    dpo_pairs.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected
                    })
                    break  # 하나의 단어당 가장 깔끔한 조합 한 개만 채택합니다.
                    
        return dpo_pairs

    def save_dataset(self, output_path="sign_dpo_dataset.json"):
        """추출된 DPO 선호도 데이터를 JSON 데이터셋으로 저장합니다."""
        dpo_data = self.find_combinations()
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=2)
            
        print(f"🎉 DPO 데이터셋 생성 완료: {output_path}")
        print(f"📊 총 생성된 학습 데이터 쌍: {len(dpo_data)}개")

if __name__ == "__main__":
    generator = DPODatasetGenerator()
    generator.save_dataset()