# config.py
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "./cache/Qwen2-1.5B-Instruct"
DATASET_PATH = "./GenAI-Hw5/Tang_training_data.json"
OUTPUT_DIR = "./output"
CKPT_DIR = "./exp1"

NUM_TRAIN_DATA = 520
EPOCHS = 1
LEARNING_RATE = 3e-4
CUTOFF_LEN = 256
MICRO_BATCH_SIZE = 2
BATCH_SIZE = 8
GRAD_ACC_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

MAX_LEN = 128
TEMPERATURE = 0.1
TOP_P = 0.3
NO_REPEAT_NGRAM_SIZE = 3