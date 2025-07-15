# test.py
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from utils import evaluate, print_results_sample
from config import (
    DEVICE, MODEL_PATH, CKPT_PATH, TEST_DATA_PATH, OUTPUT_DIR,
    MAX_LEN, TEMPERATURE, TOP_P, NO_REPEAT_NGRAM_SIZE
)


def test():
    print("\n🚀 开始 LoRA 模型测试")

    # ✅ 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "results.txt")

    # ✅ 加载 Tokenizer
    print("📚 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ✅ 加载基础模型
    print("📚 加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    )

    # ✅ 加载 LoRA 权重
    print("🔧 加载 LoRA 微调权重...")
    model = PeftModel.from_pretrained(model, CKPT_PATH, torch_dtype=torch.float32)
    model.to(DEVICE)
    print(f"✅ 模型加载完成，参数类型: {next(model.parameters()).dtype}")

    # ✅ 生成配置
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=TEMPERATURE,
        num_beams=1,
        top_p=TOP_P,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        pad_token_id=tokenizer.pad_token_id
    )

    # ✅ 读取测试数据
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_datas = json.load(f)
    print(f"📖 读取 {len(test_datas)} 个测试样例")

    # ✅ 生成结果
    results = []
    with open(output_path, "w", encoding="utf-8") as f:
        for i, data in enumerate(test_datas):
            print(f"\n➡️ 第 {i+1}/{len(test_datas)} 个样例: {data['input']}")
            try:
                output = evaluate(data["instruction"], generation_config, MAX_LEN, data["input"], tokenizer, model)
                result_line = f"{i+1}. {data['input']}{output}"
                f.write(result_line + "\n")
                print(f"生成: {output}")
                results.append({"input": data["input"], "output": output, "full_poem": result_line})
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                f.write(f"{i+1}. {data['input']} [生成失败: {e}]\n")

    print("\n🎉 测试完成！结果保存在:", output_path)
    print_results_sample(results)


if __name__ == "__main__":
    test()