import torch
from stable_baselines3 import PPO
from SignCorrectionEnv import SignCorrectionEnv

# 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 환경 및 모델 초기화
env = SignCorrectionEnv()
model = PPO("MlpPolicy", env, verbose=1, device=device)

# 학습 (최소 5만 번 이상 추천)
print(f"{device} 장치에서 학습 시작...")
model.learn(total_timesteps=50000)

# 모델 저장
model.save("sign_language_coach")
print("학습 완료 및 모델 저장 성공!")