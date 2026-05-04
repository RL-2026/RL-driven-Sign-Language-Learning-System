import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class SignCorrectionEnv(gym.Env):
    def __init__(self):
        super(SignCorrectionEnv, self).__init__()
        # 1. 정답 데이터 로드
        df = pd.read_csv('perfect_dataset.csv')
        self.target_landmarks = df.drop('label', axis=1).values
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(63,), dtype=np.float32)
        self.action_space = spaces.Discrete(6) # 0:Perfect, 1~5:손가락 지적
        
        self.state = None
        self.current_target = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 무작위로 정답 포즈 하나를 목표로 설정
        idx = np.random.randint(0, len(self.target_landmarks))
        self.current_target = self.target_landmarks[idx].astype(np.float32)
        
        # 현재 상태는 목표에 약간의 노이즈(실수)를 섞어서 시작
        self.state = self.current_target + np.random.normal(0, 0.05, 63).astype(np.float32)
        return self.state, {}

    def step(self, action):
        # [스파르타 로직] 정답과 현재 상태의 거리(오차) 계산
        dist = np.linalg.norm(self.state - self.current_target)
        
        reward = 0
        if action == 0: # 에이전트가 "잘했다"고 함
            if dist < 0.15: # 실제로 잘했을 때만 보상
                reward = 10.0
            else: # 못했는데 잘했다고 하면 큰 벌점! (이게 핵심)
                reward = -20.0
        else: # 에이전트가 "수정하라"고 함 (1~5번 액션)
            if dist >= 0.15: # 실제로 못했을 때 지적하면 보상
                reward = 5.0
            else: # 잘하고 있는데 잔소리하면 벌점
                reward = -5.0

        # 한 스텝만에 종료 (단순화)
        terminated = True
        return self.state, reward, terminated, False, {}