# test.py
import os
import json
import torch
from transformers import GenerationConfig
from config import *
from model_utils import load_model, load_tokenizer, load_finetuned_model
from evaluate import evaluate, batch_evaluate, print_results_sample, create_generation_config

def test():
    """
    测试微调后的模型
    """
    print("🚀 开始 LoRA 模型测试")
    print(f"🔧 当前模型: {CURRENT_MODEL}")
    print(f"📍 模型路径: {MODEL_PATH}")
    print(f"💾 检查点目录: {CKPT_DIR}")
    print(f"📊 测试数据: {TEST_DATA_PATH}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "test_results.txt")
    
    # 加载分词器
    tokenizer = load_tokenizer()
    
    # 加载基础模型
    base_model = load_model()
    
    # 加载微调权重
    model = load_finetuned_model(base_model, CKPT_DIR)
    
    # 创建生成配置
    generation_config = create_generation_config(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE
    )
    generation_config.pad_token_id = tokenizer.pad_token_id
    
    # 读取测试数据
    try:
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_datas = json.load(f)
        print(f"📖 读取 {len(test_datas)} 个测试样例")
    except FileNotFoundError:
        print(f"❌ 测试数据文件不存在: {TEST_DATA_PATH}")
        print("📝 使用默认测试数据")
        test_datas = [
            {"instruction": "请写一首关于春天的唐诗", "input": ""},
            {"instruction": "请写一首关于月亮的唐诗", "input": ""},
            {"instruction": "请写一首关于山水的唐诗", "input": ""}
        ]
    
    # 批量生成结果
    print("\n🎯 开始批量生成...")
    results = []
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"模型测试结果\n")
        f.write(f"模型: {CURRENT_MODEL}\n")
        f.write(f"检查点: {CKPT_DIR}\n")
        f.write(f"测试时间: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU/MPS'}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, data in enumerate(test_datas):
            print(f"\n➡️ 第 {i+1}/{len(test_datas)} 个样例")
            print(f"指令: {data['instruction']}")
            if data.get('input'):
                print(f"输入: {data['input']}")
            
            try:
                output = evaluate(
                    instruction=data["instruction"],
                    tokenizer=tokenizer,
                    model=model,
                    generation_config=generation_config,
                    max_len=MAX_LEN,
                    input_text=data.get("input", ""),
                    verbose=False
                )
                
                # 保存结果
                result_entry = {
                    "instruction": data["instruction"],
                    "input": data.get("input", ""),
                    "output": output,
                    "status": "success"
                }
                results.append(result_entry)
                
                # 写入文件
                f.write(f"样例 {i+1}:\n")
                f.write(f"指令: {data['instruction']}\n")
                if data.get('input'):
                    f.write(f"输入: {data['input']}\n")
                f.write(f"生成: {output}\n")
                f.write("-" * 30 + "\n\n")
                
                print(f"✅ 生成成功: {output}")
                
            except Exception as e:
                error_msg = f"生成失败: {str(e)}"
                print(f"❌ {error_msg}")
                
                result_entry = {
                    "instruction": data["instruction"],
                    "input": data.get("input", ""),
                    "output": "",
                    "status": "error",
                    "error": error_msg
                }
                results.append(result_entry)
                
                # 写入错误信息
                f.write(f"样例 {i+1}:\n")
                f.write(f"指令: {data['instruction']}\n")
                if data.get('input'):
                    f.write(f"输入: {data['input']}\n")
                f.write(f"错误: {error_msg}\n")
                f.write("-" * 30 + "\n\n")
    
    # 统计结果
    successful_count = sum(1 for r in results if r["status"] == "success")
    total_count = len(results)
    
    print(f"\n🎉 测试完成！")
    print(f"📊 成功率: {successful_count}/{total_count} ({successful_count/total_count*100:.1f}%)")
    print(f"💾 详细结果保存在: {output_path}")
    
    # 显示示例结果
    print_results_sample(results, num_samples=3)
    
    return results

def main():
    """
    主函数
    """
    test()

if __name__ == "__main__":
    main()