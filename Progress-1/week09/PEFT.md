# PEFT：在大模型中快速应用LoRA

> 进一步探讨如何快速地在大型预训练模型中应用LoRA
>
> 并解答可能存在的问题，包括：
>
> - `peft` 和 `lora` 之间有什么关系？
> - `get_peft_model` 怎么使用？
> - 如何知道应用 LoRA 后模型的参数变化量？
> - 如何使用 `merge_and_unload()` 合并 LoRA 权重？
> - 认识报错：`TypeError: Expected state_dict to be dict-like...`
> - 认知一个非常刁钻的 Bug：应用 LoRA 前使用 `get_peft_model()`。

## PEFT 和 LoRA 的关系

PEFT（Parameter- Efficient Fine-Tuning) 是 HF 提供的专门用于参数高效微调的工具库。LoRA 是 PEFT 支持微调的方法之一，旨在通过减少可训练参数来提高微调大模型的效率。除此之外，PEFT 还支持其他几种常见的微调方法，包括：

- Prefix-tuning：冻结原模型参数，为每一层添加可学习的前缀向量，只学习前缀参数。
- Adapter-tuning：冻结原模型参数，在模型的层与层之间插入小型的 adapter 模块，仅对 adapter 模块进行训练；

## 在大模型中应用 LoRA

### 加载预训练模型

我们以 HF 中的 transformers 库为例，加载一个预训练的 GPT2模型，其参数大小为110M。

