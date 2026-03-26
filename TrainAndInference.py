from stable_baselines3 import PPO
from SignCorrectionEnv import SignCorrectionEnv

env = SignCorrectionEnv()
# 학습량을 50,000으로 늘려 더 똑똑하게 만듭니다.
model = PPO("MlpPolicy", env, verbose=1)

print("🏋️‍♂️ 모든 알파벳에 대한 스파르타 교정법 학습 중...")

model.learn(total_timesteps=300000) 

model.save("sign_correction_ppo_model")

print("✅ 강화학습 모델(선생님) 업데이트 완료!")