#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MacBook Air M4 LoRA训练脚本 - 使用用户指定的参数配置
解决MPS设备兼容性问题
"""

# ✅ 关键修复：在导入transformers之前设置环境变量
import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
os.environ["ACCELERATE_USE_FP16"] = "false" 
os.environ["ACCELERATE_USE_BF16"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import json
import warnings
import logging
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
import transformers
from peft import PeftModel
from colorama import Fore, Style

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)

def generate_training_data(data_point):
    """
    将输入和输出文本转换为模型可读取的 tokens。
    """
    try:
        # 构建完整的输入提示词
        prompt = f"""[INST] <<SYS>>
You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
<</SYS>>

{data_point["instruction"]}
{data_point["input"]}
[/INST]"""

        # 计算用户提示词的 token 数量
        prompt_tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False,
            return_tensors=None
        )
        len_user_prompt_tokens = len(prompt_tokenized["input_ids"])

        # 将完整的输入和输出转换为 tokens
        full_text = prompt + " " + data_point["output"] + "</s>"
        full_tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding="max_length",
            return_tensors=None
        )
        
        input_ids = full_tokenized["input_ids"]
        attention_mask = full_tokenized["attention_mask"]
        
        # 创建labels，屏蔽提示词部分
        labels = input_ids.copy()
        for i in range(min(len_user_prompt_tokens, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    except Exception as e:
        print(f"数据处理错误: {e}")
        # 返回默认的数据
        return {
            "input_ids": [0] * CUTOFF_LEN,
            "labels": [-100] * CUTOFF_LEN,
            "attention_mask": [1] * CUTOFF_LEN,
        }

def evaluate(instruction, generation_config, max_len, input_text="", verbose=True):
    """
    使用 Qwen 格式生成响应。
    """
    prompt = (
        "<|im_start|>system\n你是一位擅長寫唐詩的中文助手。\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n{input_text}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=max_len,
            return_dict_in_generate=True,
            output_scores=True
        )

    output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=False)
    # 清洗输出：截断 assistant 开头后面的内容
    if "<|im_start|>assistant" in output:
        output = output.split("<|im_start|>assistant")[1]
    if "<|im_end|>" in output:
        output = output.split("<|im_end|>")[0]
    output = output.strip()

    if verbose:
        print(output)
    return output

def main():
    global tokenizer, model, device, CUTOFF_LEN
    
    print("🚀 开始MacBook Air M4 LoRA训练")
    print("✅ MPS设备环境变量已配置")
    
    # ✅ MacBook Air M4 设备配置
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ 使用設備：{device}")
    
    """ 强烈建议你尝试调整这个参数 """
    num_train_data = 520  # 设置用于训练的数据量
    
    """ 你可以（但不一定需要）更改这些超参数 """
    output_dir = "./output"
    ckpt_dir = "./exp1"
    num_epoch = 1
    LEARNING_RATE = 3e-4
    
    """ 建议不要更改此单元格中的代码 """
    cache_dir = "./cache"
    from_ckpt = False
    ckpt_name = None
    dataset_dir = "./GenAI-Hw5/Tang_training_data.json"
    logging_steps = 20
    save_steps = 65
    save_total_limit = 3
    report_to = "none"
    
    # ✅ 针对MPS设备调整的批次大小
    MICRO_BATCH_SIZE = 2  # 降低以适配MPS显存
    BATCH_SIZE = 8        # 降低以适配MPS显存
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    CUTOFF_LEN = 256
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    VAL_SET_SIZE = 0
    TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
    
    # ✅ MPS设备配置 (不使用auto device_map)
    device_map = None  # MPS不支持auto device_map
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
    
    # 创建指定的输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # ✅ 加载模型 - 适配MPS设备
    model_path = "./cache/Qwen2-1.5B-Instruct"
    
    print("📚 正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,  # MPS使用float32
        device_map=device_map
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ 模型参数类型: {next(model.parameters()).dtype}")
    
    # 根据 from_ckpt 标志，从 checkpoint 加载模型权重
    if from_ckpt:
        model = PeftModel.from_pretrained(model, ckpt_name)
    
    # ✅ 对于MPS设备，跳过量化预处理（因为我们使用float32）
    # model = prepare_model_for_kbit_training(model)  # 注释掉，因为我们不使用量化
    
    # 使用 LoraConfig 配置 LORA 模型
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    
    # ✅ 将模型移动到MPS设备
    model.to(device)
    
    # 将 tokenizer 的填充 token 设置为 0
    tokenizer.pad_token_id = 0
    
    # 加载并处理训练数据
    print(f"📊 正在加载 {num_train_data} 条训练数据...")
    with open(dataset_dir, "r", encoding="utf-8") as f:
        data_json = json.load(f)
    with open("tmp_dataset.json", "w", encoding="utf-8") as f:
        json.dump(data_json[:num_train_data], f, indent=2, ensure_ascii=False)
    
    data = load_dataset('json', data_files="tmp_dataset.json", download_mode="force_redownload")
    
    # 将训练数据分为训练集和验证集（若 VAL_SET_SIZE 大于 0）
    if VAL_SET_SIZE > 0:
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(
            generate_training_data,
            remove_columns=train_val["train"].column_names
        )
        val_data = train_val["test"].shuffle().map(
            generate_training_data,
            remove_columns=train_val["test"].column_names
        )
    else:
        train_data = data['train'].shuffle().map(
            generate_training_data,
            remove_columns=data['train'].column_names
        )
        val_data = None
    
    print("🔄 数据处理完成，开始训练配置...")
    
    # ✅ 简化的DataCollator
    def data_collator(features):
        batch = {}
        batch["input_ids"] = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return batch
    
    # ✅ 使用 Transformers Trainer 进行模型训练 - MPS兼容配置
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=50,
            num_train_epochs=num_epoch,
            learning_rate=LEARNING_RATE,
            fp16=False,  # ✅ 禁用fp16以兼容MPS
            bf16=False,  # ✅ 禁用bf16以兼容MPS
            dataloader_pin_memory=False,  # ✅ MPS兼容性
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
            output_dir=ckpt_dir,
            save_total_limit=save_total_limit,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=report_to,
            remove_unused_columns=False,  # ✅ 保持数据完整性
            max_grad_norm=1.0,  # ✅ 梯度裁剪
        ),
        data_collator=data_collator,  # ✅ 使用简化的data_collator
    )
    
    # 禁用模型的缓存功能
    model.config.use_cache = False
    
    print("✅ 开始训练...")
    
    # ✅ 开始模型训练
    try:
        trainer.train()
        print("✅ 训练完成！")
        
        # 将训练好的模型保存到指定目录
        model.save_pretrained(ckpt_dir)
        print(f"✅ 模型已保存到: {ckpt_dir}")
        
        print("\n如果上方有关于缺少键的警告，请忽略 :)")
        return True
        
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 LoRA微调成功完成！")
        print("📁 模型文件已保存到 ./exp1/ 目录")
        print("📁 Checkpoints 已保存到 ./exp1/checkpoint-* 目录")
    else:
        print("\n 训练遇到问题，请检查日志信息") 