import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 모델 로드
with open('sign_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)
rl_model = PPO.load("sign_correction_ppo_model")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
feature_names = [f'{ax}{i}' for i in range(21) for ax in ['x','y','z']]

FEEDBACK_MAP = {
    0: "PERFECT!", 1: "THUMB!", 2: "INDEX!", 3: "MIDDLE!", 4: "RING!", 5: "PINKY!"
}

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            row = [v for lm in hand_landmarks.landmark for v in [lm.x, lm.y, lm.z]]
            
            # 예측
            pred = clf_model.predict(pd.DataFrame([row], columns=feature_names))[0]
            action, _ = rl_model.predict(np.array([row], dtype=np.float32), deterministic=True)
            
            # UI 출력
            color = (0, 255, 0) if action.item() == 0 else (0, 0, 255)
            cv2.putText(image, f"Letter: {pred}", (30, 60), 1, 2, (255, 255, 255), 2)
            cv2.putText(image, f"Coach: {FEEDBACK_MAP.get(action.item(), 'ADJUST')}", (30, 110), 1, 2, color, 2)

    cv2.imshow('Final Spartan Coach', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()