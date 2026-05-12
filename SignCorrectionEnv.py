import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SignCorrectionEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 1. 데이터 로드 (npz 방식)
        gt_data  = np.load("dataset_processed/gt_sequences_ALL.npz")
        emb_data = np.load("dataset_processed/word_embeddings_ALL.npz")

        self.gt_sequences    = dict(gt_data)
        self.word_embeddings = dict(emb_data)
        self.word_list = list(self.gt_sequences.keys())

        # 2. 실제 데이터 차원 확인
        sample_key    = self.word_list[0]
        self.pose_dim  = self.gt_sequences[sample_key].shape[1]  # 134
        # ✅ 임베딩 = 첫 프레임이므로 pose_dim과 동일 (134)
        self.embed_dim = self.pose_dim  # 134

        # 3. Space 설정
        # Observation = 임베딩(134) + 현재포즈(134) = 268
        total_obs_dim = self.embed_dim + self.pose_dim  # 268
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(total_obs_dim,), dtype=np.float32
        )
        # Action = 포즈 변화량 (134차원)
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(self.pose_dim,), dtype=np.float32
        )

        print(f"✅ 환경 초기화 완료")
        print(f"   단어 수: {len(self.word_list)}개")
        print(f"   pose_dim: {self.pose_dim}, embed_dim: {self.embed_dim}")
        print(f"   observation_space: {total_obs_dim}, action_space: {self.pose_dim}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_word      = np.random.choice(self.word_list)
        self.current_frame_idx = 0
        # 현재 포즈를 첫 프레임으로 초기화
        self.current_pose = np.copy(self.gt_sequences[self.current_word][0])

        obs = np.concatenate([
            self.word_embeddings[self.current_word],  # (134,)
            self.current_pose                          # (134,)
        ]).astype(np.float32)
        return obs, {}

    def step(self, action):
        # action(변화량)을 현재 포즈에 더함
        self.current_pose      += action
        self.current_frame_idx += 1

        # Reward: 현재 포즈와 GT 포즈 사이의 거리 (가까울수록 높은 보상)
        target_pose = self.gt_sequences[self.current_word][self.current_frame_idx]
        reward      = -float(np.linalg.norm(self.current_pose - target_pose))

        # 89프레임 인덱스까지 처리 후 종료 (총 90프레임: 0~89)
        terminated = self.current_frame_idx >= 89

        obs = np.concatenate([
            self.word_embeddings[self.current_word],
            self.current_pose
        ]).astype(np.float32)

        return obs, reward, terminated, False, {}