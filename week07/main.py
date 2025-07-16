#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口点 - 微调流程管理
支持模型切换和完整的微调流程
"""

import argparse
import sys
import os
from config import CURRENT_MODEL, SUPPORTED_MODELS

def print_header():
    """打印程序头部信息"""
    print("=" * 60)
    print("🚀 大语言模型微调流程管理系统")
    print("=" * 60)
    print(f"当前模型: {CURRENT_MODEL}")
    print(f"支持的模型: {', '.join(SUPPORTED_MODELS.keys())}")
    print("=" * 60)

def run_train():
    """运行训练流程"""
    print("🏃 启动训练流程...")
    from train import main as train_main
    train_main()

def run_test():
    """运行测试流程"""
    print("🧪 启动测试流程...")
    from test import main as test_main
    test_main()

def run_interactive():
    """运行交互式对话"""
    print("💬 启动交互式对话...")
    from interactive import main as interactive_main
    interactive_main()

def switch_model(model_name):
    """切换模型"""
    if model_name not in SUPPORTED_MODELS:
        print(f"❌ 不支持的模型: {model_name}")
        print(f"支持的模型: {', '.join(SUPPORTED_MODELS.keys())}")
        return False
    
    # 读取当前配置文件
    config_path = "config.py"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 替换当前模型
    new_content = content.replace(
        f'CURRENT_MODEL = "{CURRENT_MODEL}"',
        f'CURRENT_MODEL = "{model_name}"'
    )
    
    # 写入新配置
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"✅ 模型已切换为: {model_name}")
    print("💡 请重新运行程序以应用新配置")
    return True

def show_status():
    """显示当前状态"""
    print("\n📊 当前状态:")
    print(f"当前模型: {CURRENT_MODEL}")
    
    # 检查模型文件是否存在
    model_path = SUPPORTED_MODELS[CURRENT_MODEL]["path"]
    if os.path.exists(model_path):
        print(f"✅ 模型文件存在: {model_path}")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
    
    # 检查检查点是否存在
    from config import CKPT_DIR
    if os.path.exists(CKPT_DIR):
        ckpt_files = os.listdir(CKPT_DIR)
        if any(f.endswith('.bin') or f.endswith('.safetensors') for f in ckpt_files):
            print(f"✅ 微调检查点存在: {CKPT_DIR}")
        else:
            print(f"⚠️ 微调检查点目录存在但无模型文件: {CKPT_DIR}")
    else:
        print(f"❌ 微调检查点不存在: {CKPT_DIR}")
    
    # 检查数据文件是否存在
    from config import DATASET_PATH, TEST_DATA_PATH
    if os.path.exists(DATASET_PATH):
        print(f"✅ 训练数据存在: {DATASET_PATH}")
    else:
        print(f"❌ 训练数据不存在: {DATASET_PATH}")
    
    if os.path.exists(TEST_DATA_PATH):
        print(f"✅ 测试数据存在: {TEST_DATA_PATH}")
    else:
        print(f"⚠️ 测试数据不存在: {TEST_DATA_PATH}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="大语言模型微调流程管理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py train                    # 开始训练
  python main.py test                     # 测试模型
  python main.py interactive              # 交互式对话
  python main.py switch qwen              # 切换到qwen模型
  python main.py switch llama             # 切换到llama模型
  python main.py status                   # 显示当前状态
        """
    )
    
    parser.add_argument(
        'action',
        choices=['train', 'test', 'interactive', 'switch', 'status'],
        help='要执行的操作'
    )
    
    parser.add_argument(
        'model',
        nargs='?',
        help='模型名称 (仅在switch操作时需要)'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 显示头部信息
    print_header()
    
    # 执行对应操作
    if args.action == 'train':
        run_train()
    elif args.action == 'test':
        run_test()
    elif args.action == 'interactive':
        run_interactive()
    elif args.action == 'switch':
        if not args.model:
            print("❌ 请指定要切换的模型名称")
            print(f"支持的模型: {', '.join(SUPPORTED_MODELS.keys())}")
            sys.exit(1)
        switch_model(args.model)
    elif args.action == 'status':
        show_status()

if __name__ == "__main__":
    main() 