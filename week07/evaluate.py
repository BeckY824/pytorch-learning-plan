def evaluate(instruction, generation_config, max_len, input_text="", verbose=True):
    """
    使用 QWEN 格式生成响应
    """
    prompt = (
        "<|im_start|>system\n你是一位擅长写唐诗的中文助手。\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n{input_text}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = input["input_ids"]

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=max_len,
            return_dict_in_generate=True,
            output_scores=True
        )

    output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=False)
    # 清洗输出：阶段 assistant 开头后面的内容
    if "<|im_start|>assistant" in output:
        output = output.split("<|im_start|>assistant")[1]
    if "<|im_end|>" in output:
        output = output.split("<|im_end|>")[0]
    output = output.strip()

    if verbose:
        print(output)
    return output

def print_results_sample(results, num_samples=3):
    print("\n📌 示例结果展示:")
    for i, result in enumerate(results[:num_samples]):
        print(f"\n第 {i+1} 个样例:")
        print(f"输入: {result['input']}")
        print(f"生成: {result['output']}")
        print("-" * 50)