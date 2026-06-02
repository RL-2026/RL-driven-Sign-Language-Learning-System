"""
Interactive_Render.py  ―  합성어 순차 연결 생성 및 추론 시각화 (최종 수정판)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.animation as animation
from stable_baselines3 import PPO


# ──────────────────────────────────────────────────────────
# 한글 폰트 설정 (서버 환경 대응)
# ──────────────────────────────────────────────────────────
_KO_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
_ko_font_prop = None
for _p in _KO_FONT_CANDIDATES:
    if os.path.exists(_p):
        _ko_font_prop = fm.FontProperties(fname=_p)
        fm.fontManager.addfont(_p)
        plt.rcParams['font.family'] = fm.FontProperties(fname=_p).get_name()
        break
if _ko_font_prop is None:
    _ko_font_prop = fm.FontProperties()


# ─── 스켈레톤 토폴로지 연결 정보 정의 ───────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),
    (1,11),(11,12),(12,13),
]


def _draw_frame(ax, frame_data, title="", bg_color='white'):
    """한 프레임의 관절 데이터들을 2D 평면에 매핑하여 드로잉"""
    ax.cla()
    ax.set_facecolor(bg_color)
    pts = frame_data.reshape(-1, 2)
    left, right, pose = pts[:21], pts[21:42], pts[42:]

    def _part(points, conns, color, label):
        ax.scatter(points[:, 0], -points[:, 1], c=color, s=20, zorder=3, label=label)
        for i, j in conns:
            if i < len(points) and j < len(points):
                ax.plot([points[i,0], points[j,0]],
                        [-points[i,1], -points[j,1]],
                        c=color, lw=1.2, alpha=0.7)

    _part(left,  HAND_CONNECTIONS, '#4A90D9', 'left hand')
    _part(right, HAND_CONNECTIONS, '#E05252', 'right hand')
    _part(pose,  POSE_CONNECTIONS, '#4CAF50', 'pose')
    ax.legend(loc='upper right', fontsize=7, prop=_ko_font_prop)
    ax.set_title(title, fontsize=9, fontproperties=_ko_font_prop)
    ax.axis('off')
    ax.set_aspect('equal')


def _smooth_transition(seq_a, seq_b, blend_frames=5):
    """두 동작의 물리적 불일치 경계면을 부드럽게 보간하는 모션 선형 보간 기술"""
    if blend_frames == 0 or len(seq_a) <= blend_frames:
        return np.concatenate([seq_a, seq_b], axis=0)
    transition = np.array([
        (1 - (i+1)/(blend_frames+1)) * seq_a[-blend_frames+i]
        + ((i+1)/(blend_frames+1)) * seq_b[i]
        for i in range(blend_frames)
    ], dtype=np.float32)
    body_a = seq_a[:-blend_frames]
    body_b = seq_b[blend_frames:]
    return np.concatenate([body_a, transition, body_b], axis=0)


class CompoundSignGenerator:

    def __init__(self,
                 model_path:    str = "graps_slp_model",
                 data_dir:      str = "dataset_processed",
                 total_frames:  int = 90,
                 blend_frames:  int = 5):
        print("📦 모델 및 데이터 로드 중...")
        self.model        = PPO.load(model_path, device="cpu")
        self.total_frames = total_frames
        self.blend_frames = blend_frames

        gt_data  = np.load(os.path.join(data_dir, "gt_sequences_ALL.npz"))
        emb_data = np.load(os.path.join(data_dir, "word_embeddings_ALL.npz"))

        self.gt_sequences    = {k: v.astype(np.float32) for k, v in dict(gt_data).items()}
        self.word_embeddings = {k: v.astype(np.float32) for k, v in dict(emb_data).items()}
        self.all_keys        = list(self.gt_sequences.keys())

        self._gloss_to_keys = {}
        for k in self.all_keys:
            gloss = k.split('_')[0]
            self._gloss_to_keys.setdefault(gloss, []).append(k)
        self.unique_glosses = list(self._gloss_to_keys.keys())

        print(f"✅ 로드 완료 — {len(self.all_keys):,}개 키 / {len(self.unique_glosses):,}개 어근")

    def generate(self, word: str):
        """
        [순차적 분할 추론 아키텍처] 
        원래 성분을 가장 명확히 추출하던 원본 분해기 결과를 유지하면서,
        두 단어의 고유 궤적이 시각적으로 완벽하게 분리되어 재생되도록 유도합니다.
        """
        parts = self._decompose(word)
        n_parts = len(parts)

        if n_parts == 1:
            # 단독 단어 모드
            key, gloss = parts[0]
            print(f"✨ 단독 단어 발견 및 생성 -> [{gloss}]")
            target_emb = self.word_embeddings[key]
            target_gt  = self.gt_sequences[key]
            
            ppo_seq = self._run_ppo_engine(target_emb, target_gt[0], n_frames=self.total_frames)
            gt_parts = [{"gt_seq": target_gt, "gloss": gloss, "start": 0, "end": self.total_frames}]
            return ppo_seq, gt_parts

        else:
            # 🎯 합성어 순차 생성 모드 (45프레임씩 쪼개어 단독 의미 전달력 확보)
            key_a, gloss_a = parts[0]
            key_b, gloss_b = parts[1]
            
            print(f"🔥 순차 연결 생성 엔진 가동: [{gloss_a}] (전반) ➔ [{gloss_b}] (후반)")
            
            emb1 = self.word_embeddings[key_a]
            emb2 = self.word_embeddings[key_b]
            
            # ⏳ Phase 1: 첫 번째 단어 (예: 학생) 고유 모션 명확히 추론 (45프레임)
            phase1 = self._run_ppo_engine(emb1, self.gt_sequences[key_a][0], n_frames=45)
            
            # ⏳ Phase 2: 두 번째 단어 (예: 남자) 동작 바통 이어받기 (45프레임)
            # 단어 1의 마지막 프레임 손 위치를 초기값으로 제공하여 경계면 뇌정지를 원천 차단합니다.
            current_pose = np.copy(phase1[-1])
            phase2 = []
            for _ in range(45):
                obs = np.concatenate([emb2, current_pose]).astype(np.float32)
                action, _ = self.model.predict(obs, deterministic=True)
                current_pose += action
                phase2.append(current_pose.copy())
            phase2 = np.array(phase2, dtype=np.float32)
            
            # 두 관절의 접점을 5프레임 징검다리 보간으로 자연스럽게 매칭
            combined = _smooth_transition(phase1, phase2, blend_frames=self.blend_frames)
            
            # 90프레임 규격 크기 엄수 가드
            if len(combined) > self.total_frames:
                combined = combined[:self.total_frames]
            elif len(combined) < self.total_frames:
                pad = np.tile(combined[-1:], (self.total_frames - len(combined), 1))
                combined = np.concatenate([combined, pad], axis=0)
                
            blend_gt = self.gt_sequences[key_a] * 0.5 + self.gt_sequences[key_b] * 0.5
            
            gt_parts = [{
                "gt_seq": blend_gt,
                "gloss": f"{gloss_a} ➔ {gloss_b}",
                "start": 0,
                "end": self.total_frames
            }]
            return combined, gt_parts

    def _run_ppo_engine(self, target_emb, start_pose, n_frames):
        """지정된 프레임 규격만큼 단독 관절 변위를 추론해 나가는 가벼운 서브 루프"""
        current_pose = start_pose.copy()
        frames = [current_pose.copy()]
        for _ in range(n_frames - 1):
            obs = np.concatenate([target_emb, current_pose]).astype(np.float32)
            action, _ = self.model.predict(obs, deterministic=True)
            current_pose += action
            frames.append(current_pose.copy())
        return np.array(frames, dtype=np.float32)

    def save_gif(self, ppo_seq: np.ndarray, gt_parts: list, word: str, out_path: str, fps: int = 15):
        label = gt_parts[0]['gloss']
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"단어: {word} ({label})", fontsize=12, fontweight='bold', fontproperties=_ko_font_prop)

        # 애니메이션 화면 도화지 스케일 락
        all_pts = np.concatenate([ppo_seq.reshape(-1, 2), gt_parts[0]['gt_seq'].reshape(-1, 2)], axis=0)
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        y_min, y_max = -all_pts[:, 1].max(), -all_pts[:, 1].min()
        margin = max((x_max - x_min) * 0.05, 1.0)

        def _update(f):
            for ax in axes:
                ax.set_xlim(x_min - margin, x_max + margin)
                ax.set_ylim(y_min - margin, y_max + margin)

            # 왼쪽 화면: 순차 생성 메커니즘을 거쳐 뚜렷해진 PPO 포즈
            _draw_frame(axes[0], ppo_seq[f], f"PPO 생성 ({f+1}/{self.total_frames})", 'white')

            # 오른쪽 화면: 보간 가이드 정답선 데이터 매핑
            gt_f = min(f, len(gt_parts[0]['gt_seq']) - 1)
            _draw_frame(axes[1], gt_parts[0]['gt_seq'][gt_f], f"GT 정답 [{label}] ({gt_f+1}프레임)", 'white')
            return []

        ani = animation.FuncAnimation(fig, _update, frames=self.total_frames, interval=1000 // fps, blit=False)
        ani.save(out_path, writer='pillow', fps=fps)
        plt.close()
        print(f"✅ GIF 저장 완료: {out_path}")

    # ──────────────────────────────────────────────────────────
    # 🎯 원래 가장 완벽하게 분절하던 원본 형태소 분해 엔진 파트
    # ──────────────────────────────────────────────────────────
    def _decompose(self, word):
        matched = self._exact_match(word)
        if matched:
            print(f"✨ 완전 일치: '{matched[0]}'")
            return [(matched[0], word)]

        try:
            from konlpy.tag import Okt
            morphemes = [m for m, t in Okt().pos(word) if t in ('Noun','Verb','Adjective')]
            pairs = [(self._exact_match(m)[0], m) for m in morphemes if self._exact_match(m)]
            if len(pairs) >= 2:
                print(f"🔬 KoNLPy 분리: {[g for _,g in pairs[:2]]}")
                return self._sort_by_order(word, pairs[:2])
        except ImportError:
            pass

        glosses = self._greedy_cover(word, n=2)
        if glosses:
            pairs = [(self._gloss_to_keys[g][0], g) for g in glosses]
            print(f"🧩 Greedy 분해: {glosses}")
            return self._sort_by_order(word, pairs)

        raise ValueError(f"'{word}'를 현재 사전으로 분해할 수 없습니다.")

    def _sort_by_order(self, word, pairs):
        return sorted(pairs, key=lambda x: word.find(x[1][0]) if x[1][0] in word else 999)

    def _exact_match(self, word):
        return [k for k in self.all_keys if k.split('_')[0] == word]

    def _greedy_cover(self, word, n=2):
        remaining = set(word)
        selected = []
        for _ in range(n):
            best_gloss, best_score = None, -1.0
            for gloss in self.unique_glosses:
                new_cover = set(gloss) & remaining
                if not new_cover:
                    continue
                score = len(new_cover) / len(gloss)
                if gloss[0] in word:
                    score += 0.3
                if score > best_score:
                    best_score = score
                    best_gloss = gloss
            if best_gloss is None:
                break
            selected.append(best_gloss)
            remaining -= set(best_gloss)
            if not remaining:
                break
        return selected


if __name__ == "__main__":
    gen = CompoundSignGenerator()
    while True:
        query = input("\n👉 생성할 단어 입력 (q: 종료): ").strip()
        if query.lower() in ('q', 'exit', '종료'):
            break
        if not query:
            continue
        try:
            seq, gt_parts = gen.generate(query)
            gen.save_gif(seq, gt_parts, query, f"result_{query}.gif", fps=15)
        except ValueError as e:
            print(f"⚠️  {e}")