import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from SignCorrectionEnv import SignCorrectionEnv

device = "cuda" if torch.cuda.is_available() else "cpu"
env = SignCorrectionEnv()

# 체크포인트 저장 (10만 스텝마다)
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="graps_slp"
)

model = PPO(
    "MlpPolicy", env,
    verbose=1,
    device=device,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
)

print(f"🚀 {device}에서 학습 시작...")

# ✅ 50000 → 2,000,000으로 증가
model.learn(
    total_timesteps=2_000_000,
    callback=checkpoint_callback
)

model.save("graps_slp_model")
print("✅ 모델 저장 완료")