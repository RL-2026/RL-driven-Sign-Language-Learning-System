"""
TrainAndInference.py ― 표제어 사전 기반 글로스 키 선택 에이전트 학습
"""

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from SignCorrectionEnv import SignCorrectionEnv


def main():
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "../dataset_processed"
    # 표제어 사전 경로 (프로젝트 루트에 복사해두거나 경로 조정)
    dict_xlsx = "../표제어_데이터_공공_.xlsx"

    print(f"🖥️  디바이스: {device}")

    env = SignCorrectionEnv(data_dir=data_dir, dict_xlsx=dict_xlsx)
    print(f"📊 환경 로드 완료")
    print(f"   Action space : {env.action_space}")
    print(f"   Obs dim      : {env.embed_dim}")

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints_selector/",
        name_prefix="sign_selector"
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        tensorboard_log="./tb_logs_selector/"
    )

    print("🚀 표제어 사전 기반 글로스 선택 에이전트 학습 시작...")
    model.learn(
        total_timesteps=300_000,
        callback=checkpoint_cb,
        tb_log_name="sign_selector_run"
    )

    model.save("sign_selector_model")
    print("✅ 학습 완료: sign_selector_model.zip")


if __name__ == "__main__":
    main()