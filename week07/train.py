# train.py
import transformers
from transformers import TrainingArguments, Trainer
from config import *
from model_utils import load_model, load_tokenizer, apply_lora
from data_utils import load_and_preprocess_data

def train():
    model = load_model()
    tokenizer = load_tokenizer()
    model = apply_lora(model)

    train_data, val_data = load_and_preprocess_data(tokenizer, DATASET_PATH, NUM_TRAIN_DATA)

    def collate_fn(features):
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=CKPT_DIR,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            logging_steps=20,
            save_steps=65,
            save_total_limit=3,
            fp16=False,
            dataloader_pin_memory=False,
            report_to="none",
            remove_unused_columns=False
        ),
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_fn
    )

    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(CKPT_DIR)

if __name__ == "__main__":
    train()