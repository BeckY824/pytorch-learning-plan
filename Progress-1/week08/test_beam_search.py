from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 指定模型名称
model_name = "distilgpt2"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 设置 pad_token 为 eos_token
tokenizer.pad_token = tokenizer.eos_token

# 移动模型到设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置模型为评估模式
model.eval()

# 输入文本
input_text = "Hello GPT"

# 编码输入文本，同时返回 attention_mask
inputs = tokenizer.encode_plus(input_text, return_tensors="pt", padding=True).to(device)

# 设置束宽不同的生成策略
beam_widths = [1, 3, 5]  # 使用不同的束宽

# 生成并打印结果
for beam_width in beam_widths:
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=50,
            num_beams=beam_width,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"束宽 {beam_width} 的生成结果：")
    print(generated_text)
    print('-' * 50)
