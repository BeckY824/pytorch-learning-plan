# 大语言模型微调系统

一个模块化的大语言模型微调系统，支持多种模型（Qwen、Llama等）的LoRA微调，专为唐诗生成任务设计。

## 🚀 特性

- **多模型支持**: 支持Qwen、Llama等多种大语言模型
- **模块化设计**: 清晰的代码结构，易于维护和扩展
- **MPS支持**: 针对苹果M系列芯片优化，支持MPS加速
- **交互式对话**: 提供实时交互界面测试模型效果
- **统一入口**: 通过main.py统一管理所有操作

## 📁 文件结构

```
├── main.py              # 主入口点，统一管理所有操作
├── config.py            # 配置文件，支持多模型配置
├── model_utils.py       # 模型工具，加载模型、分词器等
├── data_utils.py        # 数据处理工具，支持多模型数据格式
├── train.py             # 训练脚本
├── test.py              # 测试脚本
├── evaluate.py          # 评估工具，支持多模型生成
├── interactive.py       # 交互式对话界面
└── README.md            # 使用说明
```

## 🛠️ 安装要求

```bash
pip install torch transformers datasets peft accelerate
```

## ⚙️ 配置说明

### 切换模型

在 `config.py` 中修改 `CURRENT_MODEL` 变量，或使用命令行：

```bash
# 切换到qwen模型
python main.py switch qwen

# 切换到llama模型
python main.py switch llama
```

### 添加新模型

在 `config.py` 的 `SUPPORTED_MODELS` 字典中添加新模型配置：

```python
"new_model": {
    "path": "./cache/new-model",
    "chat_template": "new_template",
    "system_prompt": "系统提示词",
    "special_tokens": {
        "system_start": "系统开始标记",
        "system_end": "系统结束标记",
        # ... 其他特殊标记
    }
}
```

## 🚀 使用方法

### 1. 查看系统状态

```bash
python main.py status
```

### 2. 开始训练

```bash
python main.py train
```

### 3. 测试模型

```bash
python main.py test
```

### 4. 交互式对话

```bash
python main.py interactive
```

### 5. 模型切换

```bash
# 查看支持的模型
python main.py status

# 切换模型
python main.py switch qwen
python main.py switch llama
```

## 📊 数据格式

训练数据应为JSON格式，包含以下字段：

```json
[
    {
        "instruction": "写一首关于春天的唐诗",
        "input": "",
        "output": "春花烂漫满园香，蝶舞蜂飞绕花忙。"
    }
]
```

## 🎯 微调流程

1. **准备数据**: 将训练数据放在 `DATASET_PATH` 指定的位置
2. **配置模型**: 在 `config.py` 中设置模型路径和参数
3. **开始训练**: 运行 `python main.py train`
4. **测试效果**: 运行 `python main.py test`
5. **交互使用**: 运行 `python main.py interactive`

## 🔧 配置参数

主要配置参数说明：

```python
# 训练参数
NUM_TRAIN_DATA = 520        # 训练数据量
EPOCHS = 1                  # 训练轮数
LEARNING_RATE = 3e-4        # 学习率
CUTOFF_LEN = 256           # 序列最大长度
MICRO_BATCH_SIZE = 2       # 微批次大小
BATCH_SIZE = 8             # 批次大小

# LoRA参数
LORA_CONFIG = {
    "r": 8,                 # LoRA秩
    "lora_alpha": 16,       # LoRA缩放因子
    "lora_dropout": 0.05,   # Dropout率
    "bias": "none",         # 偏置设置
    "task_type": "CAUSAL_LM"
}

# 生成参数
MAX_LEN = 128              # 最大生成长度
TEMPERATURE = 0.1          # 温度参数
TOP_P = 0.3               # Top-P参数
```

## 🛡️ MPS支持

系统自动检测并使用MPS加速（适用于苹果M系列芯片）：

```python
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
```

## 📝 交互式对话命令

在交互模式中可用的命令：

- `quit/exit` - 退出程序
- `help` - 显示帮助
- `clear` - 清屏
- `history` - 显示对话历史
- `config` - 显示当前配置

## 🐛 常见问题

### Q: 模型加载失败
A: 检查模型路径是否正确，确保模型文件存在

### Q: 训练过程中内存不足
A: 减少 `MICRO_BATCH_SIZE` 或 `CUTOFF_LEN` 参数

### Q: 生成结果不理想
A: 调整温度参数 `TEMPERATURE` 或增加训练数据量

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## �� 许可证

MIT License 