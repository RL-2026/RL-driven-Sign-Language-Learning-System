import numpy as np
import os

# 기존 처리된 데이터셋에서 고유 단어(Gloss)만 추출
data_dir = "dataset_processed"
gt_data = np.load(os.path.join(data_dir, "gt_sequences_ALL.npz"))

# key 형식: "고민_NIA_SL_WORD0001..." -> 앞부분의 글로스만 추출
allowed_glosses = set(k.split('_')[0] for k in gt_data.keys())

with open("sign_dict.txt", "w", encoding="utf-8") as f:
    for gloss in sorted(allowed_glosses):
        f.write(f"{gloss}\n")

print(f"✅ 총 {len(allowed_glosses):,}개의 수어 사전 리스트 확보 완료!")