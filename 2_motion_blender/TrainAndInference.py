"""
train_blender.py ― GRAPS 상위 블렌딩 가중치 최적화 에이전트 학습 스크립트
"""

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from SignCorrectionEnv import SignCorrectionEnv

# 1. 디바이스 및 환경 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
env = SignCorrectionEnv(data_dir="dataset_processed")

# 2. 체크포인트 콜백 (5만 스텝마다 상위 에이전트 저장)
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints_blender/",
    name_prefix="graps_blender"
)

# 3. 상위 에이전트용 PPO 모델 정의
# 이 모델은 134차원 단어 벡터를 보고 1차원 alpha 가중치(Action)를 내뱉는 마스터 뇌가 됩니다.
model = PPO(
    "MlpPolicy", 
    env,
    verbose=1,
    device=device,
    learning_rate=3e-4,
    n_steps=1024,        # 에피소드가 1스텝만에 끝나므로 적절히 조절
    batch_size=64,
    n_epochs=5,
    tensorboard_log="./tb_logs_blender/"
)

print(f"🚀 상위 블렌딩 에이전트 학습 시작 ({device})...")

# 4. 학습 시작 (상위 차원 최적화는 구조가 가벼워 30만 스텝 정도면 충분히 수렴합니다)
model.learn(
    total_timesteps=300_000,
    callback=checkpoint_callback,
    tb_log_name="graps_blender_run"
)

# 5. 상위 전용 마스터 모델 저장
model.save("graps_blender_model")
print("✅ GRAPS 상위 블렌딩 최적화 모델 저장 완료 (graps_blender_model.zip)")