import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings
import os

# 1. 환경 설정 및 경고 차단
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 2. 모델 및 데이터 로드 ---
print("🧠 스파르타 수어 코치 시스템(A-Z 마스터) 로딩 중...")
try:
    with open('sign_model.pkl', 'rb') as f:
        clf_model = pickle.load(f)
    rl_model = PPO.load("sign_correction_ppo_model")
    df_target = pd.read_csv('perfect_dataset.csv')
    print("✅ 모든 모델 로드 완료! 카메라를 시작합니다.")
except Exception as e:
    print(f"❌ 로드 실패: {e}\n먼저 데이터 추출(1단계)과 학습(2,3단계)을 완료하세요.")
    exit()

# --- 3. 핵심 유틸리티 함수 ---

def get_relative_coords(coords):
    """모든 관절을 손목(0번) 기준으로 변환하여 위치/거리 차이 극복"""
    base_x, base_y, base_z = coords[0], coords[1], coords[2]
    relative = []
    for i in range(0, len(coords), 3):
        relative.extend([coords[i] - base_x, coords[i+1] - base_y, coords[i+2] - base_z])
    return np.array(relative)

def calculate_precision(user_coords, target_coords):
    """상대 좌표 기반 유클리드 거리로 정확도 계산 (완화된 기준)"""
    u_rel = get_relative_coords(user_coords)
    t_rel = get_relative_coords(target_coords)
    dist = np.linalg.norm(u_rel - t_rel)
    # 감도를 40으로 낮춰 웬만한 포즈에서 점수가 잘 나오게 설정
    score = max(0, 100 - (dist * 40)) 
    return round(score, 1)

def find_worst_finger(user_coords, target_coords):
    """손가락 끝(Tip) 좌표를 우선 비교하여 가장 오차가 큰 부위 진단"""
    u_rel = get_relative_coords(user_coords)
    t_rel = get_relative_coords(target_coords)
    
    # 손가락별 끝마디(Tip) 관절 인덱스
    tips = {"THUMB": 4, "INDEX": 8, "MIDDLE": 12, "RING": 16, "PINKY": 20}
    
    worst_error = -1
    worst_finger = "PERFECT"

    for name, idx in tips.items():
        # 손가락 끝 좌표 추출
        u_tip = u_rel[idx*3 : idx*3+3]
        t_tip = t_rel[idx*3 : idx*3+3]
        error = np.linalg.norm(u_tip - t_tip)
        
        if error > worst_error:
            worst_error = error
            worst_finger = name

    # 지적 기준을 0.5로 설정 (미세한 떨림은 무시)
    return worst_finger if worst_error > 0.5 else "PERFECT"

# --- 4. 메인 실행 루프 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# 정밀도 향상을 위해 모델 복잡도(model_complexity)를 1로 설정 가능 (기본값 1)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
feature_names = [f'{ax}{i}' for i in range(21) for ax in ['x', 'y', 'z']]

FEEDBACK_MAP = {0: "PERFECT!", 1: "THUMB!", 2: "INDEX!", 3: "MIDDLE!", 4: "RING!", 5: "PINKY!"}

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    # MediaPipe 성능을 위해 RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 화면에 뼈대 렌더링
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 실시간 좌표 리스트 생성
            row = np.array([v for lm in hand_landmarks.landmark for v in [lm.x, lm.y, lm.z]])
            
            # [판단 1] 분류 모델로 알파벳 예측
            input_df = pd.DataFrame([row], columns=feature_names)
            pred_letter = clf_model.predict(input_df)[0]
            
            # [판단 2] 강화학습 에이전트의 액션 선택
            action, _ = rl_model.predict(np.array([row], dtype=np.float32), deterministic=True)
            action_val = int(action)

            # [판단 3] 정답 데이터와 비교 분석
            target_data = df_target[df_target['label'] == pred_letter].iloc[0, 1:].values.astype(float)
            score = calculate_precision(row, target_data)
            worst_part = find_worst_finger(row, target_data)

            # --- 상단 상태창 UI ---
            cv2.rectangle(image, (20, 20), (580, 160), (0, 0, 0), -1)
            
            # 점수별 텍스트 색상 변경
            color = (0, 255, 0) if score > 75 else (0, 255, 255) if score > 45 else (0, 0, 255)
            
            cv2.putText(image, f"ALPHABET: {pred_letter} ({score}%)", (40, 70), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
            
            feedback = f"FIX: {worst_part} FINGER" if worst_part != "PERFECT" else "GREAT POSE!"
            cv2.putText(image, f"COACH: {feedback}", (40, 125), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)

    cv2.imshow('Final Spartan RL Coach', image)
    if cv2.waitKey(5) & 0xFF == 27: # ESC로 종료
        break

cap.release()
cv2.destroyAllWindows()