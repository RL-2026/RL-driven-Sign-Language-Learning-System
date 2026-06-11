"""
Interactive_Render_test.py  ―  GRAPS 시스템 최종 통합 버전 (Qwen DPO + PPO Blender)
"""

import os
import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.animation as animation
from stable_baselines3 import PPO
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── 한글 폰트 설정 ──────────────────────────────────────────────
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
PANEL_BG = ['#EBF5FF', '#FFF3E0', '#F0FFF4', '#FFF0F5']


# ── 헬퍼 함수 ────────────────────────────────────────────────────
def _axis_limits(seq: np.ndarray, margin_ratio: float = 0.08):
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

    def _part(points, conns, color):
        ax.scatter(points[:, 0], -points[:, 1], c=color, s=18, zorder=3)
        for i, j in conns:
            if i < len(points) and j < len(points):
                ax.plot(
                    [points[i,0], points[j,0]],
                    [-points[i,1], -points[j,1]],
                    c=color, lw=1.1, alpha=0.75
                )

    _part(left_h,  HAND_CONNECTIONS, '#4A90D9')
    _part(right_h, HAND_CONNECTIONS, '#E05252')
    _part(pose,    POSE_CONNECTIONS, '#4CAF50')

    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=8, fontproperties=_ko_font_prop)
    ax.axis('off')
    ax.set_aspect('equal')


def _normalize_len(seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(seq) >= target_len:
        return seq[:target_len].astype(np.float32)
    pad = np.tile(seq[-1:], (target_len - len(seq), 1))
    return np.concatenate([seq, pad], axis=0).astype(np.float32)


def _parse_glosses_from_response(response: str) -> list:
    """
    Qwen 출력에서 글로스 리스트를 추출합니다.

    우선순위:
    1) 'GLOSSES: 단어1 단어2' 라인 (학습된 포맷, 가장 안정적)
    2) '[단어]' 형태의 괄호 태그
    3) 공백 split fallback
    """
    # 1순위: GLOSSES: 라인
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("GLOSSES:"):
            parts = line.split(":", 1)[1].strip().split()
            parts = [p.strip() for p in parts if len(p.strip()) >= 2]
            if parts:
                return parts

    # 2순위: [단어] 괄호 태그
    bracket_matches = re.findall(r'\[([가-힣a-zA-Z]{2,})\]', response)
    if bracket_matches:
        return bracket_matches

    # 3순위: 공백 split (마지막 수단)
    first_line = response.split("\n")[0].strip()
    cleaned    = re.sub(r'[^\w가-힣\s]', ' ', first_line)
    fallback   = [g.strip() for g in cleaned.split() if len(g.strip()) >= 2]
    return fallback


class FullPipelineSignGenerator:

    def __init__(self,
                 qwen_model_path: str = "../1_llm_decomposer/best_qwen_sign_decomposer",
                 ppo_model_path:  str = "../2_motion_blender/graps_blender_model",
                 data_dir:        str = "../dataset_processed",
                 dict_path:       str = "../sign_dict.txt",
                 total_frames:    int = 90):

        print("📦 1단계: Qwen DPO 분해 에이전트 로드 중...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        self.qwen_tokenizer.padding_side = "left"
        self.qwen_tokenizer.pad_token    = self.qwen_tokenizer.eos_token

        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.qwen_model.eval()

        print("📦 2단계: PPO 블렌딩 에이전트 로드 중...")
        self.ppo_model    = PPO.load(ppo_model_path, device="cpu")
        self.total_frames = total_frames

        # 데이터셋 로드
        gt_data  = np.load(os.path.join(data_dir, "gt_sequences_ALL.npz"))
        emb_data = np.load(os.path.join(data_dir, "word_embeddings_ALL.npz"))
        self.gt_sequences    = {k: v.astype(np.float32) for k, v in dict(gt_data).items()}
        self.word_embeddings = {k: v.astype(np.float32) for k, v in dict(emb_data).items()}
        self.all_keys        = list(self.gt_sequences.keys())

        # 사전 단어 로드
        self.registered_glosses = set(k.split('_')[0] for k in self.all_keys)
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w:
                        self.registered_glosses.add(w)

        print(f"✅ 시스템 준비 완료! (사전 등록 단어 수: {len(self.registered_glosses):,}개)")

    # ── 🧠 Qwen DPO 분해 엔진 ────────────────────────────────────
    def _decompose_with_qwen(self, word: str) -> list:
        """
        학습 포맷과 동일한 프롬프트로 호출 →
        설명 전문을 출력하고 GLOSSES: 라인에서 글로스를 파싱합니다.
        """
        system_msg = (
            "너는 입력된 복합어를 우리가 보유한 수어 사전 단어 리스트에 맞추어 "
            "분해하는 수어 통역 에이전트야. "
            "분해 이유를 설명하고 마지막 줄에 반드시 'GLOSSES: 단어1 단어2' 형식으로 결과를 써."
        )
        user_msg = (
            f"다음 복합어를 수어 사전에 등록된 단어 단위로 분해하세요.\n"
            f"입력 단어: {word}"
        )
        chat_prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.qwen_tokenizer(
            chat_prompt, return_tensors="pt"
        ).to(self.qwen_model.device)

        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=128,          # 설명 포함이므로 128
                do_sample=False,
                eos_token_id=self.qwen_tokenizer.eos_token_id,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.qwen_tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        # ── 설명 전문 출력 (연구 목적 가시성 유지)
        print("\n================ [ Qwen 분해 설명 ] ================")
        print(response)
        print("====================================================\n")

        # ── GLOSSES: 라인 파싱
        predicted_glosses = _parse_glosses_from_response(response)
        predicted_glosses = [g for g in predicted_glosses if len(g) >= 2]
        if not predicted_glosses:
            predicted_glosses = [word]

        print(f"[파싱된 글로스]: {predicted_glosses}")

        # ── 3D 데이터셋 매칭
        valid_parts = []
        for gloss in predicted_glosses:
            matched_keys = [k for k in self.all_keys if k.split('_')[0] == gloss]
            if matched_keys:
                valid_parts.append((matched_keys[0], gloss))
            else:
                print(f"  ⚠️ [{gloss}] → 3D 데이터셋에 없음 (스킵)")

        return valid_parts

    # ── 🏃 수어 생성 메인 파이프라인 ──────────────────────────────
    def generate(self, word: str) -> dict:
        # 사전 직접 등재 확인
        exact_keys = [k for k in self.all_keys if k.split('_')[0] == word]
        if exact_keys:
            key    = exact_keys[0]
            gt_seq = _normalize_len(self.gt_sequences[key], self.total_frames)
            print(f"✅ 사전 직접 등재 단어: '{word}' (즉시 표출)")
            return {
                "word":        word,
                "output_seq":  gt_seq,
                "output_type": "gt",
                "gt_parts":    [{"gt_seq": gt_seq, "gloss": word}],
            }

        # 미등재 복합어 → Qwen 분해
        parts = self._decompose_with_qwen(word)

        if not parts:
            raise ValueError(f"'{word}'의 분해 결과를 데이터셋에서 찾을 수 없습니다.")

        # 단일 글로스 판정
        if len(parts) == 1:
            key, gloss = parts[0]
            ref_gt = _normalize_len(self.gt_sequences[key], self.total_frames)
            return {
                "word":        word,
                "output_seq":  ref_gt,
                "output_type": "gt",
                "gt_parts":    [{"gt_seq": ref_gt, "gloss": gloss}],
            }

        # 2개 이상 → PPO 블렌딩 (현재 앞 2개 사용)
        parts   = parts[:2]
        key_a, gloss_a = parts[0]
        key_b, gloss_b = parts[1]

        seq_a = _normalize_len(self.gt_sequences[key_a], self.total_frames)
        seq_b = _normalize_len(self.gt_sequences[key_b], self.total_frames)

        emb_a = self.word_embeddings.get(key_a, seq_a[0])
        emb_b = self.word_embeddings.get(key_b, seq_b[0])
        obs   = ((emb_a + emb_b) * 0.5).astype(np.float32)

        action, _ = self.ppo_model.predict(obs, deterministic=True)
        ratio = float(np.clip(action[0], 0.0, 1.0))

        print(f"⚖️  PPO 블렌딩 비율 → [{gloss_a}]: {ratio*100:.1f}%  |  [{gloss_b}]: {(1-ratio)*100:.1f}%")

        output_seq = seq_a * ratio + seq_b * (1.0 - ratio)
        gt_parts   = [
            {"gt_seq": seq_a, "gloss": gloss_a},
            {"gt_seq": seq_b, "gloss": gloss_b},
        ]

        return {
            "word":        word,
            "output_seq":  output_seq,
            "output_type": "ppo",
            "gt_parts":    gt_parts,
        }

    # ── 💾 GIF 저장 ───────────────────────────────────────────────
    def save_gif(self, result: dict, out_path: str, fps: int = 15):
        word        = result["word"]
        output_seq  = result["output_seq"]
        output_type = result["output_type"]
        gt_parts    = result["gt_parts"]
        n_gt        = len(gt_parts)

        n_cols = 1 + n_gt
        fig, axes = plt.subplots(1, n_cols, figsize=(6.0 * n_cols, 6))
        axes = [axes] if n_cols == 1 else list(axes)

        title_str = (
            f"입력: {word}  (Qwen 분해 + PPO 블렌딩)"
            if output_type == "ppo"
            else f"입력: {word}  (사전 직접 표출)"
        )
        fig.suptitle(
            title_str, fontsize=12, fontweight='bold',
            fontproperties=_ko_font_prop, y=1.01
        )

        ppo_xlim, ppo_ylim = _axis_limits(output_seq)
        gt_xlim,  gt_ylim  = _axis_limits(
            np.concatenate([gp["gt_seq"] for gp in gt_parts], axis=0)
        )
        left_label = "GT 직접 표출" if output_type == "gt" else "PPO 블렌딩 결과"

        def _update(f):
            _draw_frame(
                axes[0], output_seq[f],
                title=f"{left_label} ({f+1}/{self.total_frames}f)",
                bg_color='white', xlim=ppo_xlim, ylim=ppo_ylim
            )
            for i, gp in enumerate(gt_parts):
                gt_f = min(f, len(gp["gt_seq"]) - 1)
                bg   = PANEL_BG[i % len(PANEL_BG)] if n_gt > 1 else '#F5F5F5'
                _draw_frame(
                    axes[i + 1], gp["gt_seq"][gt_f],
                    title=f"구성 요소 [{gp['gloss']}]",
                    bg_color=bg, xlim=gt_xlim, ylim=gt_ylim
                )
            return []

        ani = animation.FuncAnimation(
            fig, _update, frames=self.total_frames,
            interval=1000 // fps, blit=False
        )
        ani.save(out_path, writer='pillow', fps=fps)
        plt.close()
        print(f"🎉 GIF 저장 완료: {out_path}\n")


# ── 메인 구동 루프 ──────────────────────────────────────────────
if __name__ == "__main__":
    try:
        pipeline = FullPipelineSignGenerator(
            qwen_model_path="../1_llm_decomposer/best_qwen_sign_decomposer",
            ppo_model_path="../2_motion_blender/graps_blender_model",
            data_dir="../dataset_processed",
            dict_path="../sign_dict.txt",
        )
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        exit(1)

    print("\n🖥️  [GRAPS End-to-End 통합 모니터 기동]")
    print("   Qwen이 복합어 분해 근거를 설명하고, PPO가 최적 모션 비율을 결정합니다.\n")

    while True:
        query = input("👉 단어 입력 (q: 종료): ").strip()
        if query.lower() in ('q', 'exit', '종료'):
            break
        if not query:
            continue
        try:
            res = pipeline.generate(query)
            pipeline.save_gif(res, f"result_{query}.gif", fps=15)
        except ValueError as e:
            print(f"⚠️  {e}")
        except Exception as e:
            print(f"🔥 렌더링 오류: {e}")