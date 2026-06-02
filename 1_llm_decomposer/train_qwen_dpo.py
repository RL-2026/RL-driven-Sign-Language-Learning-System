import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig

def main():
    # 1. 환경 및 디바이스 환경 변수 설정
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"📦 베이스 모델 로드 중: {model_id}")
    
    # 2. 토크나이저 및 Qwen 전용 ChatML 포맷 설정
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 3. 모델 로드 (오류 수정: load_from_pretrained -> from_pretrained)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # 4090 / A6000 하드웨어 가속 연산
        device_map="auto"            # 단일/멀티 GPU 자동 분산 배치
    )
    
    # 4. 고속 정렬(Alignment)을 위한 LoRA 아키텍처 설정
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. 이전 단계에서 생성한 dpo 데이터셋 로드
    dataset_path = "sign_dpo_dataset.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ {dataset_path} 파일이 없습니다. 생성기를 먼저 실행해주세요.")
        
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Qwen의 프롬프트 템플릿에 맞추어 데이터 포맷 정렬
    def format_dataset(example):
        formatted_prompt = f"<|im_start|>system\n너는 입력된 복합어를 우리가 보유한 수어 사전 단어 리스트에 맞추어 분해하는 똑똑한 수어 통역 에이전트야.<|im_end|>\n<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        return {
            "prompt": formatted_prompt,
            "chosen": example["chosen"] + "<|im_end|>",
            "rejected": example["rejected"] + "<|im_end|>"
        }

    train_dataset = raw_dataset.map(format_dataset)

    # 6. 하드웨어 스펙 최적화 하이퍼파라미터 세팅
    training_args = TrainingArguments(
        output_dir="./qwen_sign_dpo_results",
        per_device_train_batch_size=2,      
        gradient_accumulation_steps=4,      
        learning_rate=5e-5,                 
        logging_steps=5,
        num_train_epochs=3,                 
        bf16=True,                          
        gradient_checkpointing=True,        
        remove_unused_columns=False,
        save_strategy="epoch",
        eval_strategy="no",                 # 최신 버전에 맞게 인자명 및 값 수정
        warmup_ratio=0.1,
    )

    # 7. DPO 트레이너 결합 및 학습 가동
    print("🚀 Qwen-7B 수어 특화 강화학습(DPO Alignment)을 시작합니다...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,             
        args=training_args,
        beta=0.1,                   
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=256,
        max_length=512,
    )

    trainer.train()

    # 8. 수어 분해 특화 가중치 어댑터 저장
    output_model_dir = "./best_qwen_sign_decomposer"
    trainer.save_model(output_model_dir)
    print(f"✅ 세계 유일의 수어 분해 특화 Qwen 에이전트 훈련 및 저장 완료! ({output_model_dir})")

if __name__ == "__main__":
    main()