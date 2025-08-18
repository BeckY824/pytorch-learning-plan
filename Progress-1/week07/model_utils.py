# model_utils.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from config import MODEL_PATH, TARGET_MODULES, LORA_CONFIG, DEVICE, MODEL_CONFIG

def load_model():
    """加载基础模型"""
    print(f"📚 加载基础模型: {MODEL_PATH}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        model = model.to(DEVICE)
        print(f"✅ 基础模型加载完成，设备: {DEVICE}")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

def load_tokenizer():
    """加载分词器"""
    print(f"📚 加载分词器: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        print("✅ 分词器加载完成")
        return tokenizer
    except Exception as e:
        print(f"❌ 分词器加载失败: {e}")
        raise

def apply_lora(model):
    """应用LoRA配置到模型"""
    print("🔧 应用LoRA配置...")
    try:
        config = LoraConfig(target_modules=TARGET_MODULES, **LORA_CONFIG)
        peft_model = get_peft_model(model, config)
        print("✅ LoRA配置应用完成")
        return peft_model
    except Exception as e:
        print(f"❌ LoRA配置应用失败: {e}")
        raise

def load_finetuned_model(base_model, checkpoint_path):
    """加载微调后的模型"""
    print(f"🔧 加载微调权重: {checkpoint_path}")
    try:
        model = PeftModel.from_pretrained(base_model, checkpoint_path, torch_dtype=torch.float32)
        model = model.to(DEVICE)
        print(f"✅ 微调模型加载完成，参数类型: {next(model.parameters()).dtype}")
        return model
    except Exception as e:
        print(f"❌ 微调模型加载失败: {e}")
        raise

def build_prompt(instruction, input_text=""):
    """根据当前模型配置构建提示词"""
    tokens = MODEL_CONFIG["special_tokens"]
    system_prompt = MODEL_CONFIG["system_prompt"]
    
    if MODEL_CONFIG["chat_template"] == "qwen":
        prompt = (
            f"{tokens['system_start']}{system_prompt}{tokens['system_end']}"
            f"{tokens['user_start']}{instruction}\n{input_text}{tokens['user_end']}"
            f"{tokens['assistant_start']}"
        )
    elif MODEL_CONFIG["chat_template"] == "llama":
        prompt = (
            f"{tokens['system_start']}{system_prompt}{tokens['system_end']}"
            f"{instruction}\n{input_text}{tokens['user_end']}"
            f"{tokens['assistant_start']}"
        )
    else:
        raise ValueError(f"不支持的聊天模板: {MODEL_CONFIG['chat_template']}")
    
    return prompt

def clean_output(output):
    """清理模型输出"""
    tokens = MODEL_CONFIG["special_tokens"]
    
    # 移除assistant标记
    if tokens["assistant_start"] in output:
        output = output.split(tokens["assistant_start"])[-1]
    
    # 移除结束标记
    if tokens["assistant_end"] in output:
        output = output.split(tokens["assistant_end"])[0]
    
    return output.strip()