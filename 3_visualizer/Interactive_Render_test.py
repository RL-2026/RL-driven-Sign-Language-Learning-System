"""
Interactive_Render.py  ―  GRAPS 수어 생성 시각화

[패널 레이아웃]
  GT 등재 단어  →  [왼: GT]  [오: GT 참조]
  합성어 2개    →  [왼: PPO]  [오1: A GT]  [오2: B GT]  ← 동시 재생

[주요 수정]
  FIX-1  패널별 독립 축 스케일 — PPO가 GT 좌표 범위에 묻혀 안 보이는 문제 해결
  FIX-2  분해 우선순위 개선 — 이진 위치 분할 → KoNLPy → Greedy 순
          남학생 → 남자+학생 처럼 실제 단어 경계 우선 탐색
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.animation as animation
from stable_baselines3 import PPO


# ── 한글 폰트 ──────────────────────────────────────────────────
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


# ── 스켈레톤 토폴로지 ──────────────────────────────────────────
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

PANEL_BG     = ['#EBF5FF', '#FFF3E0', '#F0FFF4', '#FFF0F5']
PANEL_LABELS = ['①', '②', '③', '④']


def _axis_limits(seq: np.ndarray, margin_ratio: float = 0.08):
    """시퀀스 좌표 범위를 margin 포함해서 반환 (패널별 독립 스케일용)"""
    pts   = seq.reshape(-1, 2)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    x_pad = max((x_max - x_min) * margin_ratio, 1.0)
    y_pad = max((y_max - y_min) * margin_ratio, 1.0)
    return (x_min - x_pad, x_max + x_pad), (-y_max - y_pad, -y_min + y_pad)


def _draw_frame(ax, frame_data, title="", bg_color='white', xlim=None, ylim=None):
    ax.cla()
    ax.set_facecolor(bg_color)
    pts     = frame_data.reshape(-1, 2)
    left_h  = pts[:21]
    right_h = pts[21:42]
    pose    = pts[42:]

    def _part(points, conns, color, label):
        ax.scatter(points[:, 0], -points[:, 1], c=color, s=18, zorder=3, label=label)
        for i, j in conns:
            if i < len(points) and j < len(points):
                ax.plot([points[i,0], points[j,0]],
                        [-points[i,1], -points[j,1]],
                        c=color, lw=1.1, alpha=0.75)

    _part(left_h,  HAND_CONNECTIONS, '#4A90D9', '왼손')
    _part(right_h, HAND_CONNECTIONS, '#E05252', '오른손')
    _part(pose,    POSE_CONNECTIONS, '#4CAF50', '포즈')

    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(loc='upper right', fontsize=6, prop=_ko_font_prop)
    ax.set_title(title, fontsize=8, fontproperties=_ko_font_prop)
    ax.axis('off')
    ax.set_aspect('equal')


def _smooth_transition(seq_a, seq_b, blend_frames=5):
    if blend_frames == 0 or len(seq_a) <= blend_frames:
        return np.concatenate([seq_a, seq_b], axis=0)
    transition = np.array([
        (1 - (i+1)/(blend_frames+1)) * seq_a[-blend_frames+i]
        + ((i+1)/(blend_frames+1)) * seq_b[i]
        for i in range(blend_frames)
    ], dtype=np.float32)
    return np.concatenate([seq_a[:-blend_frames], transition, seq_b[blend_frames:]], axis=0)


def _normalize_len(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) >= target_len:
        return seq[:target_len].astype(np.float32)
    pad = np.tile(seq[-1:], (target_len - len(seq), 1))
    return np.concatenate([seq, pad], axis=0).astype(np.float32)


class CompoundSignGenerator:

    def __init__(self,
                 model_path:   str   = "graps_slp_model",
                 data_dir:     str   = "dataset_processed",
                 total_frames: int   = 90,
                 blend_frames: int   = 5,
                 stochastic:   bool  = False,
                 action_scale: float = None):
        print("📦 모델 및 데이터 로드 중...")
        self.model        = PPO.load(model_path, device="cpu")
        self.total_frames = total_frames
        self.blend_frames = blend_frames
        self.stochastic   = stochastic

        gt_data  = np.load(os.path.join(data_dir, "gt_sequences_ALL.npz"))
        emb_data = np.load(os.path.join(data_dir, "word_embeddings_ALL.npz"))

        self.gt_sequences    = {k: v.astype(np.float32) for k, v in dict(gt_data).items()}
        self.word_embeddings = {k: v.astype(np.float32) for k, v in dict(emb_data).items()}
        self.all_keys        = list(self.gt_sequences.keys())

        self._gloss_to_keys: dict[str, list] = {}
        for k in self.all_keys:
            gloss = k.split('_')[0]
            self._gloss_to_keys.setdefault(gloss, []).append(k)
        self.unique_glosses = sorted(self._gloss_to_keys.keys(), key=len, reverse=True)

        self.action_scale = action_scale if action_scale is not None \
                            else self._compute_gt_delta_median()
        print(f"📐 GT delta 중앙값(action_scale): {self.action_scale:.4f}")
        print(f"✅ 로드 완료 — {len(self.all_keys):,}개 키 / "
              f"{len(self.unique_glosses):,}개 어근")

    # ── 생성 공개 API ──────────────────────────────────────────
    def generate(self, word: str) -> dict:
        # ① GT 직접 사용
        exact_keys = self._exact_match(word)
        if exact_keys:
            key    = exact_keys[0]
            gt_seq = _normalize_len(self.gt_sequences[key], self.total_frames)
            print(f"✅ GT 직접 표출: '{word}'")
            return {
                "word":        word,
                "output_seq":  gt_seq,
                "output_type": "gt",
                "gt_parts":    [{"gt_seq": gt_seq, "gloss": word}],
            }

        # ② 분해 후 PPO 추론
        parts = self._decompose(word)

        if len(parts) == 1:
            key, gloss = parts[0]
            print(f"🤖 단일 PPO 추론: '{word}' → [{gloss}]")
            ppo_seq = self._run_ppo(
                self.word_embeddings[key],
                self.gt_sequences[key][0],
                self.total_frames,
            )
            ref_gt = _normalize_len(self.gt_sequences[key], self.total_frames)
            return {
                "word":        word,
                "output_seq":  ppo_seq,
                "output_type": "ppo",
                "gt_parts":    [{"gt_seq": ref_gt, "gloss": gloss}],
            }

        return self._generate_compound(word, parts)

    # ── 합성어 생성 ────────────────────────────────────────────
    def _generate_compound(self, word: str, parts: list) -> dict:
        n          = min(len(parts), 4)
        parts      = parts[:n]
        frames_per = self.total_frames // n
        glosses    = [g for _, g in parts]

        print(f"🔥 합성어 PPO 순차 생성: {' → '.join(f'[{g}]' for g in glosses)}")

        # 왼쪽 패널: PPO 순차 추론 → 보간 연결
        phases       = []
        current_pose = self.gt_sequences[parts[0][0]][0].copy()
        for idx, (key, _) in enumerate(parts):
            n_f   = frames_per if idx < n - 1 else (self.total_frames - frames_per * (n - 1))
            phase = self._run_ppo(self.word_embeddings[key], current_pose, n_f)
            phases.append(phase)
            current_pose = phase[-1].copy()

        combined   = phases[0]
        for ph in phases[1:]:
            combined = _smooth_transition(combined, ph, self.blend_frames)
        output_seq = _normalize_len(combined, self.total_frames)

        # 오른쪽 패널(들): 구성 단어 수만큼 GT 각각 준비 (total_frames 정규화)
        gt_parts = []
        for key, gloss in parts:
            gt_seq = _normalize_len(self.gt_sequences[key], self.total_frames)
            gt_parts.append({"gt_seq": gt_seq, "gloss": gloss})

        return {
            "word":        word,
            "output_seq":  output_seq,
            "output_type": "ppo",
            "gt_parts":    gt_parts,
        }

    # ── GIF 저장 ───────────────────────────────────────────────
    def save_gif(self, result: dict, out_path: str, fps: int = 15):
        word        = result["word"]
        output_seq  = result["output_seq"]
        output_type = result["output_type"]
        gt_parts    = result["gt_parts"]

        n_gt      = len(gt_parts)
        n_panels  = 1 + n_gt
        fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        # 제목
        if n_gt > 1:
            parts_str = " + ".join(f"[{gp['gloss']}]" for gp in gt_parts)
            suptitle  = f"입력: {word}   분해: {parts_str}"
        else:
            suptitle  = f"입력: {word}  ({gt_parts[0]['gloss']})"
        fig.suptitle(suptitle, fontsize=12, fontweight='bold',
                     fontproperties=_ko_font_prop, y=1.01)

        # ── FIX-1: 패널별 독립 축 스케일 ───────────────────────
        # 왼쪽(PPO/GT 출력): output_seq 기준으로만 스케일 결정
        ppo_xlim, ppo_ylim = _axis_limits(output_seq)

        # 오른쪽(구성 GT): 모든 GT를 합쳐서 공통 스케일 (비교 가능하게)
        all_gt_pts = np.concatenate([gp["gt_seq"].reshape(-1, 2) for gp in gt_parts], axis=0)
        gt_xlim, gt_ylim = _axis_limits(all_gt_pts.reshape(1, -1, 2).reshape(-1, 2).reshape(-1, 1, 2)
                                        .reshape(-1, 2)[np.newaxis].reshape(-1, 2))
        # 더 단순하게:
        gt_xlim, gt_ylim = _axis_limits(
            np.concatenate([gp["gt_seq"] for gp in gt_parts], axis=0)
        )

        left_label = "GT 직접 표출" if output_type == "gt" else "PPO 추론 생성"

        def _update(f):
            # 왼쪽: output (PPO 또는 GT)
            _draw_frame(
                axes[0], output_seq[f],
                title=f"{left_label}  ({f+1}/{self.total_frames}f)",
                bg_color='white',
                xlim=ppo_xlim, ylim=ppo_ylim,   # ← PPO 전용 스케일
            )

            # 오른쪽: 구성 GT 동시 재생 (각 패널 독립)
            for i, gp in enumerate(gt_parts):
                gt_f = min(f, len(gp["gt_seq"]) - 1)
                lbl  = PANEL_LABELS[i]
                if n_gt == 1:
                    title = f"GT 참조  [{gp['gloss']}]  ({gt_f+1}f)"
                    bg    = '#F5F5F5'
                else:
                    title = f"구성 GT {lbl} [{gp['gloss']}]  ({gt_f+1}f)"
                    bg    = PANEL_BG[i % len(PANEL_BG)]

                _draw_frame(
                    axes[i + 1], gp["gt_seq"][gt_f],
                    title=title, bg_color=bg,
                    xlim=gt_xlim, ylim=gt_ylim,  # ← GT 공통 스케일
                )
            return []

        ani = animation.FuncAnimation(
            fig, _update,
            frames=self.total_frames,
            interval=1000 // fps,
            blit=False,
        )
        ani.save(out_path, writer='pillow', fps=fps)
        plt.close()
        print(f"✅ GIF 저장 완료: {out_path}")

    # ── PPO 추론 ───────────────────────────────────────────────
    def _run_ppo(self, target_emb, start_pose, n_frames):
        current_pose = start_pose.copy()
        frames       = [current_pose.copy()]
        action_mags  = []
        nudge_count  = 0
        frozen_thresh = self.action_scale * 0.01

        for _ in range(n_frames - 1):
            obs    = np.concatenate([target_emb, current_pose]).astype(np.float32)
            action, _ = self.model.predict(obs, deterministic=not self.stochastic)
            mag    = float(np.mean(np.abs(action)))
            action_mags.append(mag)

            if mag < frozen_thresh:
                nudge  = np.random.randn(*action.shape).astype(np.float32)
                nudge /= (np.linalg.norm(nudge) + 1e-8)
                action = nudge * self.action_scale * 0.1
                nudge_count += 1

            current_pose = current_pose + action
            frames.append(current_pose.copy())

        mean_mag = float(np.mean(action_mags)) if action_mags else 0.0
        print(f"   action 평균: {mean_mag:.5f}  "
              f"nudge: {nudge_count}/{n_frames-1}회  "
              f"GT delta 기준: {self.action_scale:.4f}")
        return np.array(frames, dtype=np.float32)

    # ── 분해 엔진 ──────────────────────────────────────────────
    def _decompose(self, word: str) -> list:
        """
        우선순위:
          1. 완전 일치 (GT에 바로 있는 경우)
          2. 이진 위치 분할 — 남학생 → 남(자) | 학생
          3. KoNLPy 형태소 분석
          4. Greedy 문자 커버 (최후 수단)
        """
        # 1. 완전 일치
        exact = self._exact_match(word)
        if exact:
            return [(exact[0], word)]

        # 2. 이진 위치 분할 (FIX-2)
        result = self._binary_split(word)
        if result:
            glosses = [g for _, g in result]
            print(f"✂️  이진 분할: {glosses}")
            return result

        # 3. KoNLPy
        try:
            from konlpy.tag import Okt
            morphemes = [m for m, t in Okt().pos(word)
                         if t in ('Noun', 'Verb', 'Adjective')]
            pairs = [(self._exact_match(m)[0], m)
                     for m in morphemes if self._exact_match(m)]
            if len(pairs) >= 2:
                print(f"🔬 KoNLPy 분리: {[g for _, g in pairs[:2]]}")
                return self._sort_by_order(word, pairs[:2])
        except ImportError:
            pass

        # 4. Greedy
        glosses = self._greedy_cover(word, n=2)
        if glosses:
            pairs = [(self._gloss_to_keys[g][0], g) for g in glosses]
            print(f"🧩 Greedy 분해: {glosses}")
            return self._sort_by_order(word, pairs)

        raise ValueError(f"'{word}'를 현재 사전으로 분해할 수 없습니다.")

    def _binary_split(self, word: str) -> list:
        """
        단어를 모든 위치에서 2분할하여 사전 매칭 점수가 가장 높은 분할 반환.
        양쪽 모두 사전에 있으면 score=2, 한쪽만 있으면 score=1.
        score=0이면 None 반환 (Greedy로 넘김).
        """
        best_score  = 0
        best_result = None

        for split in range(1, len(word)):
            left  = word[:split]
            right = word[split:]

            left_keys  = self._exact_match(left)
            right_keys = self._exact_match(right)
            score      = (1 if left_keys else 0) + (1 if right_keys else 0)

            if score > best_score:
                best_score  = score
                # 매칭 안 된 쪽은 None 처리 (greedy 보완 가능)
                lk = left_keys[0]  if left_keys  else None
                rk = right_keys[0] if right_keys else None
                best_result = (score, lk, left, rk, right)

        if best_result is None or best_result[0] == 0:
            return None

        score, lk, left, rk, right = best_result

        # 양쪽 모두 매칭
        if score == 2:
            return [(lk, left), (rk, right)]

        # 한쪽만 매칭 — 매칭된 쪽 + 나머지는 Greedy로 보완
        if lk:
            # left 매칭, right 미매칭 → right 를 Greedy로
            supplement = self._greedy_cover(right, n=1)
            if supplement:
                sup_key = self._gloss_to_keys[supplement[0]][0]
                return [(lk, left), (sup_key, supplement[0])]
            return [(lk, left)]
        else:
            # right 매칭, left 미매칭 → left 를 Greedy로
            supplement = self._greedy_cover(left, n=1)
            if supplement:
                sup_key = self._gloss_to_keys[supplement[0]][0]
                return [(sup_key, supplement[0]), (rk, right)]
            return [(rk, right)]

    def _exact_match(self, word: str) -> list:
        return [k for k in self.all_keys if k.split('_')[0] == word]

    def _sort_by_order(self, word: str, pairs: list) -> list:
        return sorted(pairs, key=lambda x: word.find(x[1][0]) if x[1][0] in word else 999)

    def _greedy_cover(self, word: str, n: int = 2) -> list:
        """
        문자 집합 기반 Greedy.
        substring 완전 포함 시 보너스 점수 대폭 부여.
        """
        remaining = list(word)   # 순서 보존
        selected  = []
        for _ in range(n):
            best_gloss, best_score = None, -1.0
            for gloss in self.unique_glosses:
                # 완전 substring 포함 시 최우선
                if gloss in word:
                    score = len(gloss) + 2.0
                else:
                    new_cover = set(gloss) & set(remaining)
                    if not new_cover:
                        continue
                    score = len(new_cover) / max(len(gloss), 1)
                    if gloss[0] in remaining:
                        score += 0.3

                if score > best_score:
                    best_score = score
                    best_gloss = gloss

            if best_gloss is None:
                break
            selected.append(best_gloss)
            # 매칭된 gloss 문자를 remaining에서 제거
            rem_copy = remaining[:]
            for ch in best_gloss:
                if ch in rem_copy:
                    rem_copy.remove(ch)
            remaining = rem_copy
            if not remaining:
                break
        return selected

    def _compute_gt_delta_median(self) -> float:
        sample_keys = list(self.gt_sequences.keys())
        if len(sample_keys) > 300:
            rng = np.random.default_rng(42)
            sample_keys = rng.choice(sample_keys, size=300, replace=False).tolist()
        deltas = []
        for k in sample_keys:
            seq = self.gt_sequences[k]
            d   = np.abs(np.diff(seq, axis=0)).reshape(-1)
            deltas.append(d)
        all_d   = np.concatenate(deltas)
        nonzero = all_d[all_d > 0]
        return float(np.median(nonzero)) if len(nonzero) > 0 else 1.0


# ── 메인 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    gen = CompoundSignGenerator(
        model_path="graps_slp_model",
        data_dir="dataset_processed",
        total_frames=90,
        blend_frames=5,
        stochastic=False,
        action_scale=None,
    )

    print("\n[패널 구성]")
    print("  GT 등재 단어  →  [왼: GT]          [오: GT 참조]")
    print("  합성어 2개    →  [왼: PPO]  [오1: A GT]  [오2: B GT]  (동시 재생)")
    print("  ※ PPO 패널과 GT 패널은 각자 좌표 스케일 독립\n")

    while True:
        query = input("👉 단어 입력 (q: 종료): ").strip()
        if query.lower() in ('q', 'exit', '종료'):
            break
        if not query:
            continue
        try:
            result   = gen.generate(query)
            out_path = f"result_{query}.gif"
            gen.save_gif(result, out_path, fps=15)
        except ValueError as e:
            print(f"⚠️  {e}")