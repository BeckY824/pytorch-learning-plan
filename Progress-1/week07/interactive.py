#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式对话界面
与微调后的模型进行实时交互
"""

import os
import torch
from transformers import GenerationConfig
from config import *
from model_utils import load_model, load_tokenizer, load_finetuned_model
from evaluate import evaluate, create_generation_config

def interactive_chat():
    """交互式聊天功能"""
    print("🤖 正在初始化交互式对话系统...")
    print(f"当前模型: {CURRENT_MODEL}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"检查点: {CKPT_DIR}")
    
    # 加载模型和分词器
    try:
        tokenizer = load_tokenizer()
        base_model = load_model()
        
        # 检查是否有微调权重
        if os.path.exists(CKPT_DIR) and any(f.endswith('.bin') or f.endswith('.safetensors') for f in os.listdir(CKPT_DIR)):
            print("🔧 加载微调权重...")
            model = load_finetuned_model(base_model, CKPT_DIR)
            print("✅ 微调模型加载完成")
        else:
            print("⚠️ 未找到微调权重，使用基础模型")
            model = base_model
        
        # 创建生成配置
        generation_config = create_generation_config(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE
        )
        generation_config.pad_token_id = tokenizer.pad_token_id
        
        print("✅ 模型初始化完成")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 开始交互
    print("\n" + "="*50)
    print("🎭 欢迎使用唐诗生成交互系统")
    print("💡 输入你的指令，系统将为你生成唐诗")
    print("💡 输入 'quit' 或 'exit' 退出")
    print("💡 输入 'help' 查看帮助")
    print("💡 输入 'clear' 清屏")
    print("="*50)
    
    conversation_history = []
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n🧑 你: ").strip()
            
            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            elif user_input.lower() in ['help', '帮助']:
                print_help()
                continue
            elif user_input.lower() in ['clear', '清屏']:
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif user_input.lower() in ['history', '历史']:
                print_history(conversation_history)
                continue
            elif user_input.lower() in ['config', '配置']:
                print_config()
                continue
            elif not user_input:
                print("⚠️ 请输入有效的指令")
                continue
            
            # 生成响应
            print("🤖 正在生成...")
            try:
                response = evaluate(
                    instruction=user_input,
                    tokenizer=tokenizer,
                    model=model,
                    generation_config=generation_config,
                    max_len=MAX_LEN,
                    input_text="",
                    verbose=False
                )
                
                print(f"🤖 模型: {response}")
                
                # 保存对话历史
                conversation_history.append({
                    "user": user_input,
                    "assistant": response
                })
                
                # 限制历史记录长度
                if len(conversation_history) > 10:
                    conversation_history.pop(0)
                    
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

def print_help():
    """打印帮助信息"""
    print("\n📚 帮助信息:")
    print("quit/exit - 退出程序")
    print("help - 显示帮助")
    print("clear - 清屏")
    print("history - 显示对话历史")
    print("config - 显示当前配置")
    print("\n💡 示例指令:")
    print("- 写一首关于春天的唐诗")
    print("- 请创作一首描写月亮的诗")
    print("- 写一首关于山水的五言律诗")

def print_history(history):
    """打印对话历史"""
    if not history:
        print("📝 暂无对话历史")
        return
    
    print("\n📝 对话历史:")
    for i, item in enumerate(history, 1):
        print(f"{i}. 🧑: {item['user']}")
        print(f"   🤖: {item['assistant']}")
        print("-" * 40)

def print_config():
    """打印当前配置"""
    print(f"\n⚙️ 当前配置:")
    print(f"模型: {CURRENT_MODEL}")
    print(f"温度: {TEMPERATURE}")
    print(f"Top-P: {TOP_P}")
    print(f"最大长度: {MAX_LEN}")
    print(f"设备: {DEVICE}")

def main():
    """主函数"""
    interactive_chat()

if __name__ == "__main__":
    main() 