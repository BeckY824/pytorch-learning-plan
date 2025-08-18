# train.py
import torch
from transformers import TrainingArguments, Trainer
from config import *
from model_utils import load_model, load_tokenizer, apply_lora
from data_utils import load_and_preprocess_data, create_data_collator

def train():
    """
    执行模型微调训练
    """
    print("🚀 开始 LoRA 微调训练")
    print(f"🔧 当前模型: {CURRENT_MODEL}")
    print(f"📍 模型路径: {MODEL_PATH}")
    print(f"📊 数据集: {DATASET_PATH}")
    print(f"💾 输出目录: {CKPT_DIR}")
    
    # 加载模型和分词器
    model = load_model()
    tokenizer = load_tokenizer()
    
    # 应用LoRA配置
    model = apply_lora(model)
    
    # 加载和预处理数据
    train_data, val_data = load_and_preprocess_data(tokenizer, DATASET_PATH, NUM_TRAIN_DATA)
    
    # 创建数据收集器
    data_collator = create_data_collator()
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=20,
        save_steps=65,
        save_total_limit=3,
        fp16=False,  # 在MPS上使用float32
        dataloader_pin_memory=False,
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="no",  # 暂时不使用评估
        save_strategy="steps",
        logging_first_step=True,
        load_best_model_at_end=False,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
    )
    
    # 禁用缓存以节省内存
    model.config.use_cache = False
    
    # 开始训练
    print("🏃 开始训练...")
    try:
        trainer.train()
        print("✅ 训练完成")
        
        # 保存模型
        print(f"💾 保存模型到: {CKPT_DIR}")
        model.save_pretrained(CKPT_DIR)
        tokenizer.save_pretrained(CKPT_DIR)
        print("✅ 模型保存完成")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        raise

def main():
    """
    主函数
    """
    train()

if __name__ == "__main__":
    main()