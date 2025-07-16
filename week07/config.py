# config.py
import torch
import os

# 设备配置 (保留MPS支持)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 模型配置
SUPPORTED_MODELS = {
    "qwen": {
        "path": "./cache/Qwen2-1.5B-Instruct",
        "chat_template": "qwen",
        "system_prompt": "你是一位擅长写唐诗的中文助手。",
        "special_tokens": {
            "system_start": "<|im_start|>system\n",
            "system_end": "\n<|im_end|>\n",
            "user_start": "<|im_start|>user\n",
            "user_end": "\n<|im_end|>\n",
            "assistant_start": "<|im_start|>assistant\n",
            "assistant_end": "<|im_end|>",
            "eos_token": "<|im_end|>"
        }
    },
    "llama": {
        "path": "./cache/llama-model",
        "chat_template": "llama",
        "system_prompt": "You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。",
        "special_tokens": {
            "system_start": "[INST] <<SYS>>\n",
            "system_end": "\n<</SYS>>\n\n",
            "user_start": "",
            "user_end": "\n[/INST]",
            "assistant_start": "",
            "assistant_end": "</s>",
            "eos_token": "</s>"
        }
    }
}

# 当前使用的模型 (可以在这里切换模型)
CURRENT_MODEL = "qwen"
MODEL_CONFIG = SUPPORTED_MODELS[CURRENT_MODEL]
MODEL_PATH = MODEL_CONFIG["path"]

# 数据配置
DATASET_PATH = "./GenAI-Hw5/Tang_training_data.json"
TEST_DATA_PATH = "./GenAI-Hw5/test_data.json"

# 输出目录
OUTPUT_DIR = "./output"
CKPT_DIR = "./exp1"

# 训练参数
NUM_TRAIN_DATA = 520
EPOCHS = 1
LEARNING_RATE = 3e-4
CUTOFF_LEN = 256
MICRO_BATCH_SIZE = 2
BATCH_SIZE = 8
GRAD_ACC_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE

# LoRA配置
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# 生成参数
MAX_LEN = 128
TEMPERATURE = 0.1
TOP_P = 0.3
NO_REPEAT_NGRAM_SIZE = 3

# 创建必要的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)