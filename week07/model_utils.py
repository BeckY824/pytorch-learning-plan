# model_utils.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from config import MODEL_PATH, TARGET_MODULES, LORA_CONFIG, DEVICE

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map=None
    )
    return model.to(DEVICE)

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0
    return tokenizer

def apply_lora(model):
    config = LoraConfig(target_modules=TARGET_MODULES, **LORA_CONFIG)
    return get_peft_model(model, config)