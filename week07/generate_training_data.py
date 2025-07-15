def generate_training_data(data_point):
    """
    将输入和输出文本转换为模型可读的 tokens。
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
            truncation=True, #如果文本太长，超过模型所支持的最大长度，自动截断
            max_length=CUTOFF_LEN,
            padding=False,
            return_tensors=None
        )
        len_user_prompt_tokens = len(prompt_tokenized['input_ids'])

        # 将完整的输入和输出转换为 tokens
        full_text = prompt + " " + data_point["output"] + "</s>"
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
        print(f"数据处理错误: {e}")
        # 返回默认的数据
        return {
            "input_ids": [0] * CUTOFF_LEN,
            "labels": [-100] * CUTOFF_LEN,
            "attention_mask": [1] * CUTOFF_LEN,
        }
