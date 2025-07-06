# HuggingFace的AutoModel系列

选择合适的模型对于正确部署大型语言非常重要。`AutoModel` 提供了多种自动化工具，使得加载预训练模型变得非常简单。

## 主要的 `AutoModel` 类及其用途

| **类名**                             | **描述**                                                     | **适用任务**                     |
| ------------------------------------ | ------------------------------------------------------------ | -------------------------------- |
| `AutoModel`                          | 加载预训练的基础模型，不包含任何任务特定的头部。             | 特征提取、嵌入生成、自定义任务等 |
| `AutoModelForCausalLM`               | 加载带有因果语言建模头部的模型，适用于生成任务。             | 文本生成、对话系统、自动补全等   |
| `AutoModelForMaskedLM`               | 加载带有掩码语言建模头部的模型，适用于填空任务。             | 填空任务、句子补全、文本理解等   |
| `AutoModelForSeq2SeqLM`              | 加载适用于序列到序列任务的模型，带有编码器-解码器架构。      | 机器翻译、文本摘要、问答系统等   |
| `AutoModelForQuestionAnswering`      | 加载适用于问答任务的模型，带有专门的头部用于预测答案的起始和结束位置。 | 问答系统、信息检索等             |
| `AutoModelForTokenClassification`    | 加载用于标注任务（如命名实体识别）的模型。                   | 命名实体识别、词性标注等         |
| `AutoModelForSequenceClassification` | 加载用于序列分类任务的模型，带有分类头部。                   | 文本分类、情感分析等             |

## 选择合适的 `AutoModel`类

简单的指导原则：

- 文本生成：使用 Causal
- 填空任务：使用 Maksed
- 机器翻译、文本摘要：Seq2Seq
- 抽取式问答：QuestionAnswering
- 命名实体识别：TokenClassification
- 文本分类：SequenceClassification
- 特征提取或自定义任务：AutoModel

##  实际代码示例

### 示例1：文本生成

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定模型名称
model_name = "gpt2"

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(model_name)

# 输入文本
input_text = "Once upon a time"

# 编码输入
inputs = tokenizer(input_text, return_tensors="pt")

# 生成文本
outputs = model.generate(**inputs, max_length=50, do_sample=True, top_p=0.95, temperature=0.7)

# 解码生成的
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

