import torch
import cv2
import numpy as np
from ultralytics import YOLO
from stable_baselines3 import PPO

# 0. GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 장치: {device}")

# 1. 모델 로드
# YOLOv11 Pose 모델 로드 (GPU 가속)
yolo_model = YOLO('yolo11n-pose.pt').to(device)

# 학습된 RL 에이전트 로드
try:
    rl_agent = PPO.load("sign_language_coach", device=device)
    print("RL 에이전트 로드 완료.")
except:
    print("모델 파일을 찾을 수 없습니다. TrainAndInference.py를 먼저 실행하세요.")
    exit()

# 2. 정답 데이터 및 설정
target_pose = np.array([0.5, 0.5, 0.5], dtype=np.float32) # 예시 정답 좌표 (손목 혹은 특정 관절)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("시스템 시작... 'q'를 누르면 종료합니다.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # A. YOLOv11 Inference (GPU 사용)
    # stream=True로 설정하여 메모리 효율 최적화
    results = yolo_model(frame, device=device, verbose=False)

    for r in results:
        # 손목(Right Wrist) 또는 코(Nose) 등 특정 키포인트 추출
        # yolo11-pose 기준: 0: 코, 10: 오른손목 (모델마다 번호가 다를 수 있음)
        if r.keypoints is not None and len(r.keypoints.xyn) > 0:
            # 첫 번째 사람의 특정 관절 좌표 가져오기 (정규화된 0~1 값)
            # 여기서는 예시로 0번(코) 또는 특정 관절을 사용합니다.
            keypoint = r.keypoints.xyn[0][0].cpu().numpy() # [x, y]
            
            # RL State 구성을 위해 z축(0.0)을 임의로 추가하여 3차원 유지
            current_pos = np.array([keypoint[0], keypoint[1], 0.0], dtype=np.float32)

            # B. 강화학습 입력값(Observation) 생성
            obs = np.concatenate([current_pos, target_pose]).astype(np.float32)

            # C. RL 에이전트 추론
            action, _ = rl_agent.predict(obs, deterministic=True)

            # D. 시각화 및 피드백
            dist = np.linalg.norm(current_pos - target_pose)
            
            h, w, _ = frame.shape
            cx, cy = int(keypoint[0] * w), int(keypoint[1] * h)
            
            # 보상 상태 표시
            color = (0, 255, 0) if dist < 0.1 else (0, 0, 255)
            status_text = "EXCELLENT" if dist < 0.1 else f"MOVE: {action.round(2)}"

            # 화면 출력
            cv2.circle(frame, (cx, cy), 10, color, -1)
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # YOLO 관절 뼈대 그리기
            frame = r.plot() 

    cv2.imshow('YOLOv11 + RL Sign Coach', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()