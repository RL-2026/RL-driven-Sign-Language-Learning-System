"""
SignCorrectionEnv.py ― 입력 단어에 따른 최적 형태소 분해 및 블렌딩 비율 학습용 RL 환경
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SignCorrectionEnv(gym.Env):
    def __init__(self, data_dir="dataset_processed"):
        super().__init__()
        
        # 데이터셋 로드
        gt_data = np.load(os.path.join(data_dir, "gt_sequences_ALL.npz"))
        emb_data = np.load(os.path.join(data_dir, "word_embeddings_ALL.npz"))
        self.gt_sequences = {k: v.astype(np.float32) for k, v in dict(gt_data).items()}
        self.word_embeddings = {k: v.astype(np.float32) for k, v in dict(emb_data).items()}
        self.all_keys = list(self.gt_sequences.keys())
        
        # 1. Action Space: 형태소 A와 B를 섞을 '황금 가중치 비율' 결정 (0.0 ~ 1.0)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 2. Observation Space: 로드된 임베딩 파일의 실제 차원을 동적으로 탐색하여 설정
        sample_key = self.all_keys[0]
        self.embed_dim = self.word_embeddings[sample_key].shape[0] 
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.embed_dim,), dtype=np.float32)
        
        self.current_word = None
        self.morpheme_key_a = None
        self.morpheme_key_b = None
        self.current_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 💡 파이프ライン(Qwen 연동)에서 특정 단어 쌍을 주입한 경우 해당 키 사용, 없으면 무작위 학습
        if options and "morpheme_keys" in options:
            self.morpheme_key_a, self.morpheme_key_b = options["morpheme_keys"]
        else:
            self.morpheme_key_a = np.random.choice(self.all_keys)
            self.morpheme_key_b = np.random.choice(self.all_keys)
        
        # 💡 목표 복합어의 실제 임베딩 벡터가 주어지면 사용하고, 없으면 두 형태소의 중간 지점을 목표로 설정
        if options and "target_embedding" in options:
            self.current_obs = options["target_embedding"].astype(np.float32)
        else:
            self.current_obs = (self.word_embeddings[self.morpheme_key_a] + self.word_embeddings[self.morpheme_key_b]) * 0.5
        
        return self.current_obs, {}

    def step(self, action):
        ratio = float(action[0]) # 에이전트가 결정한 블렌딩 비율 (A 성분 가중치)
        
        # 🎯 3. 슬라이드 목표 보상 함수 R(s,a) 핵심 구현
        seq_a = self.gt_sequences[self.morpheme_key_a]
        seq_b = self.gt_sequences[self.morpheme_key_b]
        
        # [R_match: 기하학적 매칭 보상] 
        # 가중치 비율대로 통째로 블렌딩 시퀀스 생성 (점들이 개별로 찢어지지 않고 통째로 부드럽게 융합)
        blended_seq = seq_a * ratio + seq_b * (1.0 - ratio)
        
        # 인접 프레임 간 델타를 계산하여 기하학적 연속성 검증
        deltas = np.abs(np.diff(blended_seq, axis=0))
        r_match = -float(np.mean(deltas)) # 프레임 간 변위가 너무 급격하게 튀면 감점
        
        # [R_semantic: 의미론적 정확도 보상]
        # 에이전트가 결정한 비율로 생성된 가상 임베딩과 실제 목표 복합어 임베딩(current_obs) 간의 L2 거리를 계산
        blended_emb = self.word_embeddings[self.morpheme_key_a] * ratio + self.word_embeddings[self.morpheme_key_b] * (1.0 - ratio)
        r_semantic = -float(np.linalg.norm(blended_emb - self.current_obs))
        
        # [R_eff: 효율성 보상]
        # 어느 한쪽 형태소를 완전히 뭉개버리지 않고 균형감 있게 결합했는가
        
        
        # 🎯 슬라이드 최종 보상 바인딩 (α, β, γ 가중치 조절)
        reward = 1.0 * r_match + 2.0 * r_semantic + 0.5 
        
        # 가중치 결정은 단 1스텝 만에 끝나므로 항상 에피소드 종료 (Contextual Bandit)
        terminated = True 
        
        info = {"blended_sequence": blended_seq, "final_ratio": ratio}
        return self.current_obs, reward, terminated, False, info