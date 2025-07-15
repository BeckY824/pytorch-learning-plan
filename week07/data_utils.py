# data_utils.py
import json
from datasets import load_dataset
from config import CUTOFF_LEN

def generate_training_data(data_point, tokenizer):
    try:
        prompt = f"""[INST] <<SYS>>
You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
<</SYS>>

{data_point["instruction"]}
{data_point["input"]}
[/INST]"""

        prompt_tokenized = tokenizer(prompt, truncation=True, max_length=CUTOFF_LEN)
        len_user_prompt_tokens = len(prompt_tokenized["input_ids"])

        full_text = prompt + " " + data_point["output"] + "</s>"
        full_tokenized = tokenizer(
            full_text, truncation=True, max_length=CUTOFF_LEN, padding="max_length"
        )

        input_ids = full_tokenized['input_ids']
        attention_mask = full_tokenized["attention_mask"]
        labels = input_ids.copy()

        for i in range(min(len_user_prompt_tokens, len(labels))):
            labels[i] = -100

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    except:
        return {"input_ids": [0]*CUTOFF_LEN, "labels": [-100]*CUTOFF_LEN, "attention_mask": [1]*CUTOFF_LEN}

def load_and_preprocess_data(tokenizer, dataset_path, num_samples, val_size=0):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)
    with open("tmp_dataset.json", "w", encoding="utf-8") as f:
        json.dump(data_json[:num_samples], f, ensure_ascii=False)

    dataset = load_dataset("json", data_files="tmp_dataset.json", download_mode="force_redownload")

    if val_size > 0:
        split = dataset["train"].train_test_split(test_size=val_size)
        train = split["train"].map(lambda x: generate_training_data(x, tokenizer))
        val = split["test"].map(lambda x: generate_training_data(x, tokenizer))
        return train, val
    else:
        train = dataset["train"].map(lambda x: generate_training_data(x, tokenizer))
        return train, None