import json
import os
import numpy as np

MORPHEME_ROOT = "dataset/1.Training/morpheme"
TRAINING_ROOT = "dataset/1.Training"
OUTPUT_DIR = "dataset_processed"

def parse_keypoints_flat(flat_array, n_points=21, stride=3):
    """플랫 배열 [x, y, conf, x, y, conf, ...] → x,y만 추출한 리스트"""
    points = []
    for i in range(0, n_points * stride, stride):
        if i + 1 < len(flat_array):
            points.extend([flat_array[i], flat_array[i + 1]])
        else:
            points.extend([0.0, 0.0])
    return points

def load_morpheme_map():
    morpheme_map = {}
    print("📦 형태소 데이터 로드 중...")
    for root, dirs, files in os.walk(MORPHEME_ROOT):
        for file_name in files:
            if not file_name.endswith(".json"):
                continue
            file_id = file_name.replace("_morpheme.json", "")
            full_path = os.path.join(root, file_name)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    m_data = json.load(f)
                data_list = m_data.get('data', [])
                if not data_list:
                    continue
                attr_list = data_list[0].get('attributes', [])
                if not attr_list:
                    continue
                gloss = attr_list[0].get('name', 'unknown')
                if gloss != 'unknown':
                    morpheme_map[file_id] = gloss
            except Exception as e:
                print(f"  ⚠️ 형태소 파싱 오류 ({file_name}): {e}")

    print(f"✅ 형태소 맵 로드 완료: {len(morpheme_map)}개")
    return morpheme_map

def process_zone(sub_dir, morpheme_map):
    """특정 구역(01, 02...) 하나만 처리하고 저장"""
    output_gt  = os.path.join(OUTPUT_DIR, f"gt_sequences_{sub_dir}.npz")
    output_emb = os.path.join(OUTPUT_DIR, f"word_embeddings_{sub_dir}.npz")

    # 이미 처리된 구역은 스킵
    if os.path.exists(output_gt) and os.path.exists(output_emb):
        print(f"⏭️  구역 {sub_dir} 이미 처리됨, 스킵")
        return

    current_path = os.path.join(TRAINING_ROOT, sub_dir)
    if not os.path.isdir(current_path):
        print(f"❌ 구역 {sub_dir} 폴더 없음")
        return

    print(f"\n📂 구역 처리 중: {sub_dir}")
    gt_sequences   = {}
    word_embeddings = {}
    processed_count = 0
    skipped_count   = 0

    for word_dir in sorted(os.listdir(current_path)):
        word_path = os.path.join(current_path, word_dir)
        if not os.path.isdir(word_path):
            continue

        current_file_id = word_dir  # 폴더명 자체가 ID

        if current_file_id not in morpheme_map:
            skipped_count += 1
            continue

        gloss = morpheme_map[current_file_id]
        json_files = sorted([f for f in os.listdir(word_path) if f.endswith(".json")])
        if not json_files:
            continue

        current_seq = []
        for f_name in json_files:
            f_path = os.path.join(word_path, f_name)
            try:
                with open(f_path, 'r', encoding='utf-8') as kf:
                    k_data = json.load(kf)

                people = k_data.get('people', {})

                # 왼손 21개 (x,y) = 42차원
                left_hand  = parse_keypoints_flat(people.get('hand_left_keypoints_2d',  []), 21, 3)
                # 오른손 21개 (x,y) = 42차원
                right_hand = parse_keypoints_flat(people.get('hand_right_keypoints_2d', []), 21, 3)
                # 포즈 25개 (x,y) = 50차원
                pose       = parse_keypoints_flat(people.get('pose_keypoints_2d',       []), 25, 3)

                # 총 134차원 (42 + 42 + 50)
                frame_data = left_hand + right_hand + pose
                if len(frame_data) == 134:
                    current_seq.append(frame_data)

            except Exception as e:
                print(f"  ⚠️ 키포인트 오류 ({f_name}): {e}")

        if current_seq:
            # 90프레임 규격화 (134차원)
            final_seq = np.zeros((90, 134), dtype=np.float32)
            len_seq = min(len(current_seq), 90)
            final_seq[:len_seq] = current_seq[:len_seq]

            save_key = f"{gloss}_{current_file_id}"
            gt_sequences[save_key]    = final_seq
            # ✅ 랜덤값 대신 첫 프레임을 임베딩으로 사용 (항상 동일한 값 보장)
            word_embeddings[save_key] = final_seq[0]  # shape: (134,)

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"  ✅ {processed_count}번째 처리 완료 ({gloss})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(output_gt,  **gt_sequences)
    np.savez(output_emb, **word_embeddings)
    print(f"  💾 저장 완료: {processed_count}개 처리, {skipped_count}개 스킵")

def merge_all_zones():
    """모든 구역 npz 합치기"""
    print("\n🔗 전체 구역 합치는 중...")
    merged_gt  = {}
    merged_emb = {}

    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith("gt_sequences_") and f.endswith(".npz") and "ALL" not in f:
            zone = f.replace("gt_sequences_", "").replace(".npz", "")
            gt_data  = np.load(os.path.join(OUTPUT_DIR, f))
            emb_path = os.path.join(OUTPUT_DIR, f"word_embeddings_{zone}.npz")
            emb_data = np.load(emb_path)

            merged_gt.update(dict(gt_data))
            merged_emb.update(dict(emb_data))
            print(f"  📥 구역 {zone}: {len(gt_data.files)}개")

    np.savez(os.path.join(OUTPUT_DIR, "gt_sequences_ALL.npz"),   **merged_gt)
    np.savez(os.path.join(OUTPUT_DIR, "word_embeddings_ALL.npz"), **merged_emb)
    print(f"\n🎉 합본 저장 완료: 총 {len(merged_gt)}개")
    print(f"   저장 형태: (90프레임, 134차원) — 왼손42 + 오른손42 + 포즈50")

if __name__ == "__main__":
    morpheme_map = load_morpheme_map()

    # 처리할 구역 목록 — 새 구역 추가 시 여기에만 추가하면 됨
    # 이미 처리된 구역은 자동으로 스킵됨
    target_zones = ["01", "02", "03", "04", "05", "06"]

    for zone in target_zones:
        process_zone(zone, morpheme_map)

    merge_all_zones()