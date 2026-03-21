import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SignCorrectionEnv(gym.Env):
    def __init__(self):
        super(SignCorrectionEnv, self).__init__()

        # 1. State (Observation Space) 정의
        # [현재 손가락 21개 관절 x,y,z (63)] + [목표 손가락 21개 관절 x,y,z (63)] = 총 126개
        # 좌표값은 정규화된 0.0 ~ 1.0 사이의 값으로 가정합니다.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(126,), dtype=np.float32
        )

        # 2. Action Space 정의
        # 에이전트가 각 관절(21개)을 어느 방향으로 움직여야 할지 지시 (dx, dy, dz)
        # 총 63개의 연속적인 움직임 값을 출력합니다.
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(63,), dtype=np.float32
        )

        # 3. 초기화 데이터 설정
        self.target_pose = np.random.rand(63).astype(np.float32) # 실제로는 정석 수어 데이터 로드
        self.current_pos = np.zeros(63, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 사용자가 카메라 앞에 나타났을 때의 초기 랜덤 위치 (학습용 다양성 확보)
        self.current_pos = np.random.uniform(0.3, 0.7, size=63).astype(np.float32)
        
        # 현재 상태와 목표 상태를 합쳐서 반환
        observation = np.concatenate([self.current_pos, self.target_pose])
        return observation, {}

    def step(self, action):
        # 1. 에이전트의 Action을 현재 위치에 적용
        self.current_pos = np.clip(self.current_pos + action, 0.0, 1.0)
        
        # 2. 보상(Reward) 계산 - 수어 교정의 핵심 로직
        # 현재 모든 관절과 목표 관절 사이의 유클리드 거리 합 계산
        dist = np.linalg.norm(self.current_pos - self.target_pose)
        
        # 기본 보상: 거리가 멀수록 마이너스
        reward = -dist 
        
        # 보너스 보상: 목표치에 매우 근접했을 때 (정밀도 유도)
        if dist < 0.1:
            reward += 5.0
        if dist < 0.05:
            reward += 10.0
            
        # 3. 종료 조건 (성공 판정)
        # 전체 관절 오차가 매우 작아지면 학습 에피소드 종료
        terminated = bool(dist < 0.03)
        
        # 4. 다음 관측값 생성
        observation = np.concatenate([self.current_pos, self.target_pose])
        
        # Gymnasium 인터페이스: obs, reward, terminated, truncated, info
        return observation, reward, terminated, False, {}

    def render(self):
        # 필요 시 시각화 로직 추가 (보통 pipe.py에서 cv2로 처리)
        pass