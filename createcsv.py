import cv2
import mediapipe as mp
import os
import csv
import string

# 1. 경로 설정 (반드시 본인의 폴더 구조 확인)
DATASET_DIR = 'dataset/asl_alphabet_train/asl_alphabet_train' 
TARGET_CLASSES = list(string.ascii_uppercase)
CSV_FILE = 'perfect_dataset.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# CSV 헤더 정의
header = ['label'] + [f'{ax}{i}' for i in range(21) for ax in ['x', 'y', 'z']]

print("🚀 [1단계] A-Z 데이터 추출 시작...")

with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    total_count = 0
    for alphabet in TARGET_CLASSES:
        folder_path = os.path.join(DATASET_DIR, alphabet)
        if not os.path.exists(folder_path): continue

        print(f"[{alphabet}] 처리 중...", end=' ')
        image_files = [img for img in os.listdir(folder_path) if img.lower().endswith(('.jpg', '.png'))][:200]
        
        count = 0
        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue
            
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row = [alphabet] + [val for lm in hand_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
                    writer.writerow(row)
                    count += 1
                    total_count += 1
        print(f"성공 ({count}개)")

hands.close()
print(f"✅ 총 {total_count}개 데이터 저장 완료!")