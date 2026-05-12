import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO

# 손 관절 연결 정보 (MediaPipe 기준 21개)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

# 포즈 연결 정보 (COCO 기준 25개 중 주요 연결)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),
    (1,11),(11,12),(12,13),
]

def draw_frame(ax, frame_data, title=""):
    """한 프레임을 ax에 그리기"""
    ax.cla()
    pts = frame_data.reshape(-1, 2)  # (67, 2)

    left  = pts[:21]    # 왼손
    right = pts[21:42]  # 오른손
    pose  = pts[42:]    # 포즈

    def draw_part(points, connections, color, label):
        ax.scatter(points[:, 0], -points[:, 1], c=color, s=20, zorder=3, label=label)
        for i, j in connections:
            if i < len(points) and j < len(points):
                ax.plot(
                    [points[i, 0], points[j, 0]],
                    [-points[i, 1], -points[j, 1]],
                    c=color, linewidth=1.2, alpha=0.7
                )

    draw_part(left,  HAND_CONNECTIONS, 'blue',  '왼손')
    draw_part(right, HAND_CONNECTIONS, 'red',   '오른손')
    draw_part(pose,  POSE_CONNECTIONS, 'green', '포즈')

    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    ax.set_aspect('equal')

def make_gif(sequence, gt_seq, target_key, file_name, fps=15):
    """sequence와 gt_seq를 나란히 GIF로 저장"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"단어: {target_key}", fontsize=13)

    # 전체 좌표 범위 계산 (축 고정용)
    all_pts = np.concatenate([
        np.array(sequence).reshape(-1, 2),
        gt_seq.reshape(-1, 2)
    ], axis=0)
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = -all_pts[:, 1].max(), -all_pts[:, 1].min()
    margin = 30

    def update(frame_idx):
        for ax in axes:
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)

        draw_frame(axes[0], sequence[frame_idx],   title=f"PPO 생성  (프레임 {frame_idx+1}/90)")
        draw_frame(axes[1], gt_seq[frame_idx],     title=f"GT 정답   (프레임 {frame_idx+1}/90)")

        for ax in axes:
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)

        return []

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(sequence),
        interval=1000 // fps,
        blit=False
    )

    ani.save(file_name, writer='pillow', fps=fps)
    plt.close()
    print(f"✅ GIF 저장 완료: {file_name}  ({len(sequence)}프레임, {fps}fps)")

# ───────────────────────────────────────────
print("📦 모델 및 데이터 로드 중...")
model    = PPO.load("graps_slp_model")
gt_data  = np.load("dataset_processed/gt_sequences_ALL.npz")
emb_data = np.load("dataset_processed/word_embeddings_ALL.npz")

gt_sequences    = dict(gt_data)
word_embeddings = dict(emb_data)
all_words = list(gt_sequences.keys())

print(f"✅ 총 {len(all_words)}개 단어 로드 완료")
print(f"📋 샘플: {all_words[:5]}")

while True:
    search_query = input("\n👉 생성할 단어 입력 (q: 종료): ").strip()
    if search_query.lower() in ['q', 'exit']:
        break

    matched_keys = [k for k in all_words if search_query in k]
    if not matched_keys:
        print(f"⚠️  '{search_query}'가 포함된 단어를 찾을 수 없습니다.")
        continue

    if len(matched_keys) > 1:
        print(f"🔍 검색 결과 ({len(matched_keys)}개): {matched_keys[:5]}")
        print(f"   → '{matched_keys[0]}' 선택")

    target_key = matched_keys[0]
    print(f"✨ '{target_key}' 90프레임 생성 중...")

    # PPO로 시퀀스 생성
    emb          = word_embeddings[target_key]   # (134,)
    current_pose = np.copy(emb).astype(np.float32)
    sequence     = [current_pose.copy()]

    for _ in range(89):
        obs = np.concatenate([emb, current_pose]).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        current_pose += action
        sequence.append(current_pose.copy())

    gt_seq    = gt_sequences[target_key]         # (90, 134)
    file_name = f"result_{search_query}.gif"

    make_gif(sequence, gt_seq, target_key, file_name, fps=15)