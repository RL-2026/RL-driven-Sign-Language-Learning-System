import json
import os
import numpy as np

# ── 💡 절대경로 자동 계산 로직 (실행 위치가 어디든 안전하게 폴더 추적) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # 0_data_pipeline 폴더 위치
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # 프로젝트 루트 (~/git/)

MORPHEME_ROOT = os.path.join(PROJECT_ROOT, "5_dataset/morpheme")
TRAINING_ROOT = os.path.join(PROJECT_ROOT, "5_dataset")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "dataset_processed")

print(f"📁 상위 프로젝트 경로: {PROJECT_ROOT}")
print(f"🔍 읽어올 데이터 경로: {TRAINING_ROOT}")
print(f"💾 내보낼 전처리 경로: {OUTPUT_DIR}")


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
    print("\n📦 형태소 데이터 구조 정밀 분석 및 전수 로드 중...")
    if not os.path.exists(MORPHEME_ROOT):
        raise FileNotFoundError(f"❌ 형태소 폴더를 찾을 수 없습니다: {MORPHEME_ROOT}")
        
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
                
                # 🔥 [핵심 수정 1] 0번 인덱스 고정을 풀고, 리스트 내 모든 형태소(Morpheme) 단어 수집
                gloss_list = []
                for attr in attr_list:
                    gloss_name = attr.get('name', 'unknown').strip()
                    if gloss_name != 'unknown' and gloss_name not in gloss_list:
                        gloss_list.append(gloss_name)
                
                # 파일 ID 하나에 매칭된 모든 형태소 배열 저장
                if gloss_list:
                    morpheme_map[file_id] = gloss_list
                    
            except Exception as e:
                print(f"  ⚠️ 형태소 파싱 오류 ({file_name}): {e}")

    print(f"✅ 형태소 멀티 맵 빌드 완료: {len(morpheme_map):,}개 파일의 하위 구조선 확보")
    return morpheme_map

def process_zone(sub_dir, morpheme_map):
    """특정 구역(01, 02...) 내 모든 폴더의 형태소를 확장하여 npz로 압축 저장"""
    output_gt  = os.path.join(OUTPUT_DIR, f"gt_sequences_{sub_dir}.npz")
    output_emb = os.path.join(OUTPUT_DIR, f"word_embeddings_{sub_dir}.npz")

    # 🔥 [핵심 수정 2] 데이터셋 확장을 위해 기존의 강제 스킵 로직 완전 비활성화
    # if os.path.exists(output_gt) and os.path.exists(output_emb):
    #     print(f" Richmond ⏭️  구역 {sub_dir} 이미 처리됨, 스킵")
    #     return

    current_path = os.path.join(TRAINING_ROOT, sub_dir)
    if not os.path.isdir(current_path):
        print(f"❌ 구역 {sub_dir} 폴더 없음")
        return

    print(f"\n📂 구역 처리 및 형태소 다중 매핑 가동: {sub_dir}")
    gt_sequences   = {}
    word_embeddings = {}
    processed_video_count = 0
    total_expanded_count = 0
    skipped_count   = 0

    for word_dir in sorted(os.listdir(current_path)):
        word_path = os.path.join(current_path, word_dir)
        if not os.path.isdir(word_path):
            continue

        current_file_id = word_dir  # 폴더명 자체가 ID

        if current_file_id not in morpheme_map:
            skipped_count += 1
            continue

        # 🔥 [핵심 수정 3] 1:1 단어가 아닌 쪼개진 모든 형태소 리스트 호출
        gloss_list = morpheme_map[current_file_id]
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

            # 🔥 [핵심 수정 4] 하나의 영상을 쪼개진 모든 형태소 키값으로 복사 복제 (데이터 증강)
            # 예: '여학생' 영상 하나를 [여학생_ID], [여자_ID], [학생_ID] 키포인트로 전부 파생 등록
            for gloss in gloss_list:
                save_key = f"{gloss}_{current_file_id}"
                gt_sequences[save_key]    = final_seq
                word_embeddings[save_key] = final_seq[0]
                total_expanded_count += 1

            processed_video_count += 1
            if processed_video_count % 100 == 0:
                print(f"  ✅ 영상 {processed_video_count}개 ➔ 누적 형태소 확장 데이터셋: {total_expanded_count}개 생성")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(output_gt,  **gt_sequences)
    np.savez(output_emb, **word_embeddings)
    print(f"  💾 저장 완료 [구역 {sub_dir}]: 원본 {processed_video_count}개 ➔ 형태소 결합 증강 {total_expanded_count}개 완료 (스킵: {skipped_count}개)")

def merge_all_zones():
    """모든 구역 npz 합치기"""
    print("\n🔗 증강 완료된 전체 구역 데이터 병합 가동...")
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
            print(f"  📥 구역 {zone} (형태소 확장본): {len(gt_data.files):,}개 시퀀스 병합")

    np.savez(os.path.join(OUTPUT_DIR, "gt_sequences_ALL.npz"),   **merged_gt)
    np.savez(os.path.join(OUTPUT_DIR, "word_embeddings_ALL.npz"), **merged_emb)
    print(f"\n🎉 대규모 형태소 확장 합본 생성 최종 완료: 총 {len(merged_gt):,}개 데이터 확보!")
    print(f"   위치: {os.path.join(OUTPUT_DIR, 'gt_sequences_ALL.npz')}")

if __name__ == "__main__":
    morpheme_map = load_morpheme_map()

    # 01부터 04구역까지 전체 리스트 동기화 순회
    target_zones = ["01", "02", "03", "04"]

    for zone in target_zones:
        process_zone(zone, morpheme_map)

    merge_all_zones()