# evaluate.py
import torch
from transformers import GenerationConfig
from model_utils import build_prompt, clean_output

def evaluate(instruction, tokenizer, model, generation_config, max_len, input_text="", verbose=True):
    """
    使用当前模型配置生成响应，支持多种模型格式
    """
    # 构建提示词
    prompt = build_prompt(instruction, input_text)
    
    if verbose:
        print(f"📝 提示词: {prompt}")
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_ids = inputs["input_ids"]
    
    # 生成响应
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=max_len,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 解码输出
    output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=False)
    
    # 清理输出
    cleaned_output = clean_output(output)
    
    if verbose:
        print(f"🎯 生成结果: {cleaned_output}")
    
    return cleaned_output

def batch_evaluate(instructions, tokenizer, model, generation_config, max_len, input_texts=None, verbose=True):
    """
    批量评估多个指令
    """
    if input_texts is None:
        input_texts = [""] * len(instructions)
    
    if len(instructions) != len(input_texts):
        raise ValueError("指令数量和输入文本数量不匹配")
    
    results = []
    for i, (instruction, input_text) in enumerate(zip(instructions, input_texts)):
        if verbose:
            print(f"\n➡️ 处理第 {i+1}/{len(instructions)} 个样例")
        
        try:
            output = evaluate(instruction, tokenizer, model, generation_config, max_len, input_text, verbose)
            results.append({
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "status": "success"
            })
        except Exception as e:
            error_msg = f"生成失败: {str(e)}"
            if verbose:
                print(f"❌ {error_msg}")
            results.append({
                "instruction": instruction,
                "input": input_text,
                "output": "",
                "status": "error",
                "error": error_msg
            })
    
    return results

def print_results_sample(results, num_samples=3):
    """
    打印结果样例
    """
    print(f"\n📌 示例结果展示 (显示前 {num_samples} 个):")
    successful_results = [r for r in results if r.get("status") == "success"]
    
    if not successful_results:
        print("❌ 没有成功的生成结果")
        return
    
    for i, result in enumerate(successful_results[:num_samples]):
        print(f"\n第 {i+1} 个样例:")
        print(f"指令: {result['instruction']}")
        if result['input']:
            print(f"输入: {result['input']}")
        print(f"生成: {result['output']}")
        print("-" * 50)

def create_generation_config(temperature=0.1, top_p=0.3, no_repeat_ngram_size=3):
    """
    创建生成配置
    """
    return GenerationConfig(
        do_sample=True,
        temperature=temperature,
        num_beams=1,
        top_p=top_p,
        no_repeat_ngram_size=no_repeat_ngram_size
    )