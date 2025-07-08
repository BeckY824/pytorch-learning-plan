#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MacBook Air M4 LoRA模型测试脚本
测试训练完成的唐诗生成模型
"""

# ✅ MPS设备环境变量配置
import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
os.environ["ACCELERATE_USE_FP16"] = "false" 
os.environ["ACCELERATE_USE_BF16"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import json
import warnings
import logging
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel

def evaluate(instruction, generation_config, max_len, input_text="", verbose=True):
    """
    使用 Qwen 格式生成响应。
    """
    prompt = (
        "<|im_start|>system\n你是一位擅長寫唐詩的中文助手。\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n{input_text}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=max_len,
            return_dict_in_generate=True,
            output_scores=True
        )

    output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=False)
    # 清洗输出：截断 assistant 开头后面的内容
    if "<|im_start|>assistant" in output:
        output = output.split("<|im_start|>assistant")[1]
    if "<|im_end|>" in output:
        output = output.split("<|im_end|>")[0]
    output = output.strip()

    if verbose:
        print(output)
    return output

def main():
    global tokenizer, model, device
    
    print("🚀 开始LoRA模型测试")
    print("✅ MPS设备环境变量已配置")
    
    # ✅ MacBook Air M4 设备配置
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ 使用设备：{device}")
    
    # 配置参数
    model_name = "Qwen/Qwen2-1.5B-Instruct"
    model_path = "./cache/Qwen2-1.5B-Instruct"  # 本地模型路径
    ckpt_name = "./exp1"  # 训练好的LoRA权重路径
    test_data_path = "GenAI-Hw5/Tang_testing_data.json"
    output_dir = "./output"
    output_path = os.path.join(output_dir, "results.txt")
    
    # 生成参数
    max_len = 128
    temperature = 0.1
    top_p = 0.3
    no_repeat_ngram_size = 3
    seed = 42
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("📚 正在加载基础模型...")
    
    # ✅ 加载 tokenizer - 适配MacBook Air M4
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # ✅ 加载基础模型 - 使用float32，不使用量化
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # MPS使用float32
        low_cpu_mem_usage=True,
        device_map=None  # MPS不支持auto device_map
    )
    
    print("🔧 正在加载LoRA权重...")
    
    # ✅ 加载微调后的LoRA权重
    model = PeftModel.from_pretrained(
        model, 
        ckpt_name,
        torch_dtype=torch.float32
    )
    
    # 将模型移动到MPS设备
    model.to(device)
    
    print(f"✅ 模型加载完成，参数类型: {next(model.parameters()).dtype}")
    
    # 设置生成配置
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        num_beams=1,
        top_p=top_p,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # 读取测试数据集
    print("📖 正在读取测试数据...")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_datas = json.load(f)
    
    print(f"✅ 找到 {len(test_datas)} 个测试样例")
    
    # 开始测试生成
    print("🎭 开始生成唐诗...")
    results = []
    
    # 对每个测试样例生成预测，并保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        for (i, test_data) in enumerate(test_datas):
            print(f"\n处理第 {i+1}/{len(test_datas)} 个样例...")
            print(f"输入: {test_data['input']}")
            
            try:
                predict = evaluate(
                    test_data["instruction"], 
                    generation_config, 
                    max_len, 
                    test_data["input"], 
                    verbose=False
                )
                
                result_line = f"{i+1}. " + test_data["input"] + predict
                f.write(result_line + "\n")
                
                print(f"生成: {predict}")
                print(f"完整结果: {result_line}")
                
                results.append({
                    "input": test_data["input"],
                    "output": predict,
                    "full_poem": result_line
                })
                
            except Exception as e:
                error_msg = f"第 {i+1} 个样例生成失败: {e}"
                print(f"❌ {error_msg}")
                f.write(f"{i+1}. {test_data['input']} [生成失败: {e}]\n")
    
    print(f"\n🎉 测试完成！")
    print(f"📁 结果已保存到: {output_path}")
    print(f"✅ 成功生成 {len([r for r in results if 'output' in r])} 首唐诗")
    
    # 显示几个示例结果
    print("\n📝 部分生成示例:")
    for i, result in enumerate(results[:3]):  # 显示前3个结果
        print(f"\n示例 {i+1}:")
        print(f"输入: {result['input']}")
        print(f"生成: {result['output']}")
        print("-" * 50)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎊 LoRA模型测试成功完成！")
        else:
            print("\n💡 测试遇到问题，请检查日志")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 