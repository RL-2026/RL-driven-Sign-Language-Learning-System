from stable_baselines3 import PPO
from SignCorrectionEnv import SignCorrectionEnv
import gymnasium as gym

gym.envs.registration.register(
    id='SignCorrection-v0',
    entry_point='SignCorrectionEnv:SignCorrectionEnv',
)

env = gym.make('SignCorrection-v0')
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)

print("🏋️‍♂️ [3단계] 지옥의 콤보 훈련 중... (2~5분 소요)")
model.learn(total_timesteps=100000)
model.save("sign_correction_ppo_model")
print("✅ 코치 훈련 완료!")