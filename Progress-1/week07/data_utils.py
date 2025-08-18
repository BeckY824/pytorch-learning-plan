# data_utils.py
import json
import os
import torch
from datasets import load_dataset
from config import CUTOFF_LEN, MODEL_CONFIG

def generate_training_data(data_point, tokenizer):
    """
    将输入和输出文本转换为模型可读的tokens，支持多种模型格式
    """
    try:
        # 根据模型类型构建提示词
        tokens = MODEL_CONFIG["special_tokens"]
        system_prompt = MODEL_CONFIG["system_prompt"]
        
        if MODEL_CONFIG["chat_template"] == "qwen":
            prompt = (
                f"{tokens['system_start']}{system_prompt}{tokens['system_end']}"
                f"{tokens['user_start']}{data_point['instruction']}\n{data_point['input']}{tokens['user_end']}"
                f"{tokens['assistant_start']}"
            )
            full_text = prompt + data_point["output"] + tokens["eos_token"]
        elif MODEL_CONFIG["chat_template"] == "llama":
            prompt = (
                f"{tokens['system_start']}{system_prompt}{tokens['system_end']}"
                f"{data_point['instruction']}\n{data_point['input']}{tokens['user_end']}"
                f"{tokens['assistant_start']}"
            )
            full_text = prompt + data_point["output"] + tokens["eos_token"]
        else:
            raise ValueError(f"不支持的聊天模板: {MODEL_CONFIG['chat_template']}")

        # 计算用户提示词的token数量
        prompt_tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False,
            return_tensors=None
        )
        len_user_prompt_tokens = len(prompt_tokenized['input_ids'])

        # 将完整的输入和输出转换为tokens
        full_tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding="max_length",
            return_tensors=None
        )

        input_ids = full_tokenized['input_ids']
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
        print(f"❌ 数据处理错误: {e}")
        # 返回默认的数据
        return {
            "input_ids": [0] * CUTOFF_LEN,
            "labels": [-100] * CUTOFF_LEN,
            "attention_mask": [1] * CUTOFF_LEN,
        }

def load_and_preprocess_data(tokenizer, dataset_path, num_samples, val_size=0):
    """
    加载并预处理训练数据
    """
    print(f"📖 加载数据集: {dataset_path}")
    print(f"📊 数据量: {num_samples}, 验证集比例: {val_size}")
    
    try:
        # 读取原始数据
        with open(dataset_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)
        
        # 限制数据量
        selected_data = data_json[:num_samples]
        
        # 创建临时数据集文件
        tmp_dataset_path = "tmp_dataset.json"
        with open(tmp_dataset_path, "w", encoding="utf-8") as f:
            json.dump(selected_data, f, ensure_ascii=False, indent=2)
        
        # 加载数据集
        dataset = load_dataset("json", data_files=tmp_dataset_path, download_mode="force_redownload")
        
        # 预处理数据
        if val_size > 0:
            # 分割训练集和验证集
            split = dataset["train"].train_test_split(test_size=val_size)
            train_dataset = split["train"].map(lambda x: generate_training_data(x, tokenizer))
            val_dataset = split["test"].map(lambda x: generate_training_data(x, tokenizer))
            
            print(f"✅ 数据预处理完成 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
            
            # 清理临时文件
            if os.path.exists(tmp_dataset_path):
                os.remove(tmp_dataset_path)
            
            return train_dataset, val_dataset
        else:
            # 只有训练集
            train_dataset = dataset["train"].map(lambda x: generate_training_data(x, tokenizer))
            
            print(f"✅ 数据预处理完成 - 训练集: {len(train_dataset)}")
            
            # 清理临时文件
            if os.path.exists(tmp_dataset_path):
                os.remove(tmp_dataset_path)
            
            return train_dataset, None
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        # 清理临时文件
        if os.path.exists("tmp_dataset.json"):
            os.remove("tmp_dataset.json")
        raise

def create_data_collator():
    """创建数据收集器"""
    def collate_fn(features):
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        }
    return collate_fn