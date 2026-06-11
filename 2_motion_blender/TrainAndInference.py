"""
TrainAndInference.py ― GRAPS 상위 블렌딩 가중치 최적화 에이전트 학습 스크립트
"""

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from SignCorrectionEnv import SignCorrectionEnv

def main():
    # 1. 디바이스 및 환경 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 연산 디바이스 상태: {device}")
    
    # 데이터셋 경로 재확인 및 환경 선언
    data_dir = "../dataset_processed"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        
    env = SignCorrectionEnv(data_dir=data_dir)
    print(f"📊 환경 로드 완료 (단어 임베딩 동적 탐색 차원: {env.embed_dim})")

    # 2. 체크포인트 콜백 (5만 스텝마다 모델 중간 저장)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints_blender/",
        name_prefix="graps_blender"
    )

    # 3. 상위 에이전트용 PPO 모델 정의
    # 1스텝 만에 보상이 나오는 Bandit 문제이므로 batch_size와 n_steps 밸런스 정렬
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        n_steps=1024,        
        batch_size=64,
        n_epochs=5,
        tensorboard_log="./tb_logs_blender/"
    )

    print(f"🚀 상위 블렌딩 에이전트 강화학습 시작 ({device})...")

    # 4. 학습 시작
    model.learn(
        total_timesteps=300_000,
        callback=checkpoint_callback,
        tb_log_name="graps_blender_run"
    )

    # 5. 최종 마스터 모델 저장
    model.save("graps_blender_model")
    print("✅ GRAPS 상위 블렌딩 최적화 모델 최종 저장 완료 (graps_blender_model.zip)")

if __name__ == "__main__":
    main()