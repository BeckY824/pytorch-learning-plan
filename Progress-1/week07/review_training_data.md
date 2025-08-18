# Training data reviews

## 数据结构分析

每条数据均为标准的 SFT（supervised Fine-Tuning）格式：

```json
{
  "instruction": "以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。",
  "input": "白日依山盡，黃河入海流。",
  "output": "欲窮千里目，更上一層樓。"
}
```

**数据格式特点：**

- 统一指令 + 独立首句输入 + 唐诗全诗输出
- 共 25007 条数据
- 每条 output 为完整诗句（4～8句），可视作 "诗歌补全" 任务

## Token 统计与内存评估

平均每条：instruction ~ 20 tokens + input ~ 10 tokens + output ~ 50 tokens ≈ **80 tokens**

总 tokens ：25000 ✖️ 80 = 2000000 tokens (200万)

**评估：**实际占用内存仅1.4mb，每条文本都很短。用于微调非常合适，但对大模型是"低负载任务"

## 训练价值分析

**优点**：

- 数据格式清晰、内容高质量（真实唐诗），可用于诗歌生成或语言补全任务。
- 能帮助模型学会：从一两句古诗中推测其"风格/主题/语义走向"，是一种 风格迁移 + 指令微调 的复合任务。

**局限**：

- 规模仅200万 token，对于 >=1B 参数量的 LLM 来说远远不够（通常微调需要 > 10M token 才有效果显著）
- 所有 instruction 都是相同句式，缺乏多样性（建议后续做 instruction mix）

## 设备需求与训练建议

| **项目**   | **参数建议**                              |
| ---------- | ----------------------------------------- |
| 模型       | Qwen2-1.5B HF 格式                        |
| 微调方法   | LoRA (推荐使用 peft + transformers)       |
| 序列长度   | 256~512 tokens（唐诗足够短）              |
| Batch Size | 4~8（根据 MPS 显存，建议 batch=4 起试）   |
| Epoch      | 5~20（由于数据少，epoch 可多）            |
| 优化器     | AdamW，学习率 2e-4 起试                   |
| 精度       | MPS 推荐用 float32 或 bf16（稳定性更高）  |
| 训练时间   | 预计 30~60 分钟以内（取决于 batch/epoch） |

## 进阶建议

1. 加入额外任务：
   - "改写诗风" 或 "风格模仿" : 例如加入 instruction：模仿李白风格续写下面诗句
   - 添加现代语解释任务
2. 指令多样化
   - 把instruction改写为多种表达方式，增加泛化能力：
     - 请继续补全这首诗
     - 请写出这首唐诗的后续内容
     - 请续写这首古诗...
3. 生成格式控制：
   - 添加 系统提示系统 prompt , 用于结构化训练

## 训练参数分析

```python
# 训练参数
NUM_TRAIN_DATA = 520
EPOCHS = 1
LEARNING_RATE = 3e-4
CUTOFF_LEN = 256
MICRO_BATCH_SIZE = 2
BATCH_SIZE = 8
GRAD_ACC_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE

# LoRA配置
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# 生成参数
MAX_LEN = 128
TEMPERATURE = 0.1
TOP_P = 0.3
NO_REPEAT_NGRAM_SIZE = 3
```

| **参数名**           | **含义与分析**                                               |
| -------------------- | ------------------------------------------------------------ |
| NUM_TRAIN_DATA = 250 | 表示当前只挑选了520条样本用于训练（不是全部的25007条），训练非常小 |
| EPOCHS = 1           | 只训练一轮，基本等于 warm-up 水平训练，建议改为 3～10轮      |
| LEARNING_RATE = 3e-4 | 偏高（Lora一般用 2e-4 ~ 1e-4），由于EPOCHS少，没大问题。若 EPOCH增多， 改为 2e-4 |
| CUTOFF_LEN = 256     | 非常适合。唐诗每条约 60～100 tokens，256 足够覆盖 input + output，且节省内存 |
| MICRO_BATCH_SIZE = 2 | 每个 MPS 步只处理 2 个样本，是安全设置。若有更大显存，可尝试 4 |
| BATCH_SIZE = 8       | 表示一个优化步中，包含8个样本                                |
| GRAD_ACC_STEPS = 4   | 8/2 = 4, 配合显存不足、又想要大 batch，有效                  |

## **LoRA 配置分析**

```python
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

| **参数**                | **分析**                                                     |
| ----------------------- | ------------------------------------------------------------ |
| TARGET_MODULES          | 针对 Qwen2 架构选择了 Q/K/V 和 FFN 各关键路径模块。尤其 gate_proj, down_proj, up_proj 是 FFN 结构中的主要通路。 |
| r = 8                   | LoRA 低秩矩阵秩为8， 标准设置，推荐                          |
| lora_alpha = 16         | 通常设置为 r * 2                                             |
| lora_dropout = 0.05     | 训练时对 adapter dropout，适度正则化，防止过拟合。           |
| bias = "none"           | 只对 adapter 做训练，主模型参数不动。非常适合快速微调        |
| task_type = "CAUSAL_LM" | 对应生成任务（诗歌续写），正确                               |

## 生成参数分析

```python
MAX_LEN = 128
TEMPERATURE = 0.1
TOP_P = 0.3
NO_REPEAT_NGRAM_SIZE = 3	
```

| **参数**                 | **分析**                                                     |
| ------------------------ | ------------------------------------------------------------ |
| MAX_LEN = 128            | 生成最大长度，适合唐诗续写任务（大多诗句在 100 tokens 内）   |
| TEMPERATURE = 0.1        | 非常低，会导致模型生成非常保守甚至死板的输出（建议改为 0.7～1.0） |
| TOP_P = 0.3              | 比较收敛的 top-p 的采样策略，值略低                          |
| NO_REPEAT_NGRAM_SIZE = 3 | 禁止 3gram 重复，有助于避免无意义重复                        |