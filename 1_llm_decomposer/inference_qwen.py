import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_trained_decomposer():
    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    adapter_dir = "./best_qwen_sign_decomposer"
    
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"❌ 학습된 어댑터 폴더({adapter_dir})를 찾을 수 없습니다. 학습이 완료되었는지 확인하세요.")

    print("📦 베이스 모델 및 토크나이저 로드 중 (12GB VRAM 최적화)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print("🪄 학습된 수어 분해 특화 LoRA 가중치 결합 중...")
    # 베이스 Qwen 모델 위에 우리가 DPO로 학습시킨 어댑터를 얹습니다.
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    
    return model, tokenizer

def decompose_word(word, model, tokenizer):
    """테스트 단어를 입력받아 Qwen이 분해한 리스트를 반환합니다."""
    prompt = f"다음 복합어를 수어 사전에 등록된 단어 단위로 분해하세요.\n입력 단어: {word}"
    
    # Qwen ChatML 포맷 적용
    chat_prompt = (
        f"<|im_start|>system\n너는 입력된 복합어를 우리가 보유한 수어 사전 단어 리스트에 맞추어 분해하는 똑똑한 수어 통역 에이전트야.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False # 정밀한 정답 추출을 위해 Greedy decoding 사용
        )
    
    # 입력 프롬프트 부분을 제외한 모델의 순수 답변만 추출
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    try:
        # JSON 형태의 문자열 파싱 (예: '["가족", "회의"]' -> python list)
        decomposed_list = json.loads(response)
        return decomposed_list
    except:
        # 혹시나 예외 텍스트가 나왔을 때의 예외 처리
        return response

if __name__ == "__main__":
    # 1. 모델 로드
    model, tokenizer = load_trained_decomposer()
    print("✨ 수어 분해 에이전트 준비 완료!\n")
    
    # 2. 테스트해 볼 복합어 리스트 (사전 기반)
    test_words = ["가족회의", "치과의사", "달서구수어통역센터", "생각못하다", "칠백오십만"]
    
    print("🧪 [테스트 시작] Qwen 모델의 분해 성능 검증")
    print("="*50)
    for word in test_words:
        result = decompose_word(word, model, tokenizer)
        print(f" 입력 복합어 : {word}")
        print(f" 🤖 Qwen 분해 : {result}")
        print(f" 데이터 타입 : {type(result)}")
        print("-"*50)