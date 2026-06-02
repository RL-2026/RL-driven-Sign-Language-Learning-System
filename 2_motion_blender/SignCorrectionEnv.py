"""
GRAPSBlendingEnv.py ― 입력 단어에 따른 최적 형태소 분해 및 블렌딩 비율 학습용 RL 환경
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
        # 예: Action이 0.7이면 남아 70% + 학업 30%로 자연스럽게 선형 블렌딩
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 2. Observation Space: 현재 입력된 합성어 단어의 임베딩 상태 (예: 남학생 Word2Vec/KoBERT 벡터 차원)
        # 하위 포즈 dim(134)이 아니라, 순수 단어 의미 차원만 다룹니다.
        self.embed_dim = 134 
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.embed_dim,), dtype=np.float32)
        
        self.current_word = None
        self.morpheme_key_a = None
        self.morpheme_key_b = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 🎯 실제 실험 시에는 여기서 '남학생' 같은 데이터셋 내 사전에 없는 복합어 샘플링 유도
        # 임시로 랜덤 명사 쌍 매칭 시뮬레이션
        self.morpheme_key_a = np.random.choice(self.all_keys)
        self.morpheme_key_b = np.random.choice(self.all_keys)
        
        # 현재 에피소드의 가상 목표 복합어 임베딩 (두 의미의 중간 공간)
        self.current_obs = (self.word_embeddings[self.morpheme_key_a] + self.word_embeddings[self.morpheme_key_b]) * 0.5
        
        return self.current_obs, {}

    def step(self, action):
        ratio = float(action[0]) # 에이전트가 결정한 블렌딩 비율 (A 성분 가중치)
        
        # 🎯 3. 슬라이드 목표 보상 함수 R(s,a) 핵심 구현
        
        # [R_match: 기하학적 매칭 보상] 
        # 두 수어가 섞였을 때 손 정렬이 부드럽게 유지되는지 (동작 간 급격한 좌표 튐 패널티)
        seq_a = self.gt_sequences[self.morpheme_key_a]
        seq_b = self.gt_sequences[self.morpheme_key_b]
        
        # 가중치 비율대로 통째로 블렌딩 시퀀스 생성 (점들이 개별로 찢어지지 않고 통째로 부드럽게 융합)
        blended_seq = seq_a * ratio + seq_b * (1.0 - ratio)
        
        # 인접 프레임 간 델타를 계산하여 기하학적 연속성 검증
        deltas = np.abs(np.diff(blended_seq, axis=0))
        r_match = -float(np.mean(deltas)) # 프레임 간 변위가 너무 급격하게 튀면 감점
        
        # [R_semantic: 의미론적 정확도 보상]
        # 에이전트가 섞은 비율이 목표 복합어의 의미 공간 코사인 유사도와 얼마나 일치하는가
        # (여기서는 예시로 타겟 임베딩 공간과의 거리를 역산)
        r_semantic = -float(np.abs(ratio - 0.5)) # 가상의 타겟 의미 중심점(0.5)에서 멀어지면 감점
        
        # [R_eff: 효율성 보상]
        # 어느 한쪽 형태소를 완전히 뭉개버리지 않고 균형감 있게 결합했는가
        r_eff = -float((ratio - 0.5) ** 2)
        
        # 🎯 슬라이드 최종 보상 바인딩 (α, β, γ 가중치 조절)
        reward = 1.0 * r_match + 2.0 * r_semantic + 0.5 * r_eff
        
        # 가중치 결정은 단 1스텝 만에 끝나므로 항상 에피소드 종료
        terminated = True 
        
        info = {"blended_sequence": blended_seq, "final_ratio": ratio}
        return self.current_obs, reward, terminated, False, info