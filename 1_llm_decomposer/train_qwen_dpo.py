"""
train_qwen_dpo.py
chosen/rejected 포맷: 설명형 텍스트 + 'GLOSSES: 단어1 단어2' 라인
"""

import torch
import torch.distributed.fsdp as fsdp
import os

# 🔥 핫픽스 1: Hugging Face 최신 버전 import 버그 방지
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

# 🔥 핫픽스 2: PyTorch FSDP2 호환성 버그 방지
if not hasattr(fsdp, "FSDPModule"):
    class DummyFSDPModule:
        pass
    fsdp.FSDPModule = DummyFSDPModule

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"📦 베이스 모델 로드 중: {model_id}")

    # ── 토크나이저
    # 설명형 chosen이 여러 줄이므로 model_max_length 넉넉히 설정
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token        = tokenizer.eos_token
    tokenizer.padding_side     = "left"
    tokenizer.model_max_length = 768

    # ── 모델 (bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    # ── LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 데이터셋 로드
    dataset_path = "sign_dpo_dataset.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"❌ {dataset_path} 없음. make_dpo_dataset.py 먼저 실행하세요."
        )

    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    def format_dataset(example):
        """
        chosen/rejected가 이제 설명형 텍스트이므로 그대로 붙이면 됩니다.
        예시 chosen:
            교통사고는 [교통]과 [사고]로 분해됩니다.
            교통: 차량이나 사람이 도로를 이동하는 행위
            사고: 예상치 못하게 발생한 나쁜 사건
            GLOSSES: 교통 사고
        """
        system_msg = (
            "너는 입력된 복합어를 우리가 보유한 수어 사전 단어 리스트에 맞추어 "
            "분해하는 수어 통역 에이전트야. "
            "분해 이유를 설명하고 마지막 줄에 반드시 'GLOSSES: 단어1 단어2' 형식으로 결과를 써."
        )
        formatted_prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return {
            "prompt":   formatted_prompt,
            "chosen":   example["chosen"]   + "<|im_end|>",
            "rejected": example["rejected"] + "<|im_end|>",
        }

    train_dataset = raw_dataset.map(format_dataset)

    # ── DPO 설정
    training_args = DPOConfig(
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
        eval_strategy="no",
        warmup_ratio=0.1,
        beta=0.1,
        save_safetensors=False,

    )

    print("🚀 Qwen-7B LoRA DPO 학습 시작...")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    output_model_dir = "./best_qwen_sign_decomposer"
    trainer.save_model(output_model_dir)
    print(f"✅ 학습 완료 및 저장: {output_model_dir}")


if __name__ == "__main__":
    main()