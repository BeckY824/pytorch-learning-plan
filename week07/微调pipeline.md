# 模块化微调大模型

一共包括6个核心pipeline，划分为以下 **从数据到部署**

## 1.数据准备

| **步骤**     | **内容**                                                     |
| ------------ | ------------------------------------------------------------ |
| 收集数据     | 从 JSON、TXT 或第三方来源构造监督训练样本                    |
| 格式化数据   | 每条样本转成 {instruction, input, output} 结构               |
| tokenization | 使用 tokenizer 对 prompt + output 进行编码，生成 input_ids 和 labels |
| 清洗/填充    | 控制最大长度、掩码、padding 等处理逻辑                       |

核心函数：

```python
def generate_training_data(data_point):
 		# 将单条数据打包为 input_ids / labels / attention_mask
```

## 2. 模型加载 (Model Loading)

| **步骤**       | **内容**                                   |
| -------------- | ------------------------------------------ |
| 加载预训练模型 | 如 Qwen/Qwen2、LLaMA、Baichuan、ChatGLM 等 |
| 设置推理精度   | CPU/MPS 用 float32，GPU 可用 FP16/BF16     |
| 加载 tokenizer | 保证模型输入正确对应字典表                 |

示例代码：

```python
model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)
```

## 3. 准备 LoRA

| **步骤**           | **内容**                           |
| ------------------ | ---------------------------------- |
| 加载 LoRA config   | 指定 r 值、dropout、target modules |
| 加载 PEFT 模型     | 使用 get_peft_model                |
| 准备参数冻结       | 大模型冻结，只有 adapter 学习      |
| 量化预处理（可选） | 用于 GPU 上的 qLoRA，但 MPS 不适用 |

核心代码：

```python
model = get_peft_model(base_model, LoraConfig(...))
```

## 4. 训练流程

| **步骤**          | **内容**                                       |
| ----------------- | ---------------------------------------------- |
| 构建 Dataset      | Huggingface Dataset 或自定义 DataLoader        |
| 定义 DataCollator | 拼 batch 时对齐输入维度                        |
| 设置训练器        | transformers.Trainer 或 Accelerate             |
| 配置超参数        | batch size、learning rate、logging、checkpoint |

核心模块：

```python
trainer = transformers.Trainer(...)
trainer.train()
```

## 5. 评估与验证

| **步骤**   | **内容**                                |
| ---------- | --------------------------------------- |
| 生成函数   | 自定义 evaluate()，输入 prompt 输出文本 |
| 可选验证集 | 在训练过程中评估性能指标（loss）        |
| 生成样本   | 检查模型是否学会“指令 + 生成”的结构     |

核心函数：

```python
def evaluate(instruction, generation_config, max_len, input_text="")
```

## 6. 保存与部署

| **步骤**         | **内容**                                           |
| ---------------- | -------------------------------------------------- |
| 保存模型         | .save_pretrained() 保存 LoRA adapter               |
| 合并权重（可选） | 可将 adapter 合并入 base 模型                      |
| 加载推理         | 用微调模型继续 generate() 推理                     |
| 部署方案         | Gradio / FastAPI / Streamlit / Hugging Face Spaces |

示例：

```python
model.save_pretrained("my_checkpoint")
tokenizer.save_pretrained("my_checkpoint")
```

