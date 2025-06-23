# 如何训练一个LLM

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250623143832313.png" alt="image-20250623143832313" style="zoom:50%;" />

训练一个完整的LLM需要经过三个阶段——Pretrain、SFT和RLHF。

## Pretrain

是训练LLM最核心也是工程量最大的第一步。目前主流的LLM采用了 Decoder-Only 类的 GPT 架构，它们的预训练任务也都是——因果语言模型。

因果语言模型建模，通过给出上文要求模型预测下一个 token 来进行训练。 LLM 的预训练同传统预训练模型的核心差异在于，预训练的体量和资源消耗。

LLM往往需要使用更大规模的预训练语料。分布式训练框架也成为LLM训练必不可少的组成部分。分布式训练框架的核心思路是数据并行和模型并行。

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250623144641865.png" alt="image-20250623144641865" style="zoom:50%;" />

> 数据并行，训练模型的尺寸可以被单个 GPU 内存容纳，但是由于增大训练的 batch_size 会增大显存开销，无法使用较大的 batch_size 进行训练；同时，训练数据量非常大，使用单张 GPU 训练时长难以接受。
>
> 当LLM扩大到上百亿参数，单张 GPU 内存往往就无法存放完整的模型参数。可以讲模型拆分到多个 GPU 上，每个 GPU 上存放不同的层或不同的部分，从而实现模型并行。

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250623144811407.png" alt="image-20250623144811407" style="zoom:50%;" />

Deepspeed是主流的分布式训练框架。核心策略是 ZeRo 和 CPU-offload。ZeRo 是一种显存优化的数据并行方案，其核心思想是优化数据并行时每张卡的显存占用，从而实现对更大规模模型的支持。

预训练数据的处理与清晰也是LLM预训练的一个重要环节。诸多研究证明，预训练数据的质量往往比体量更加重要。

- 文档准备
- 语料过滤
- 语料去重

目前，已有很多经过处理的高质量预训练语料和专用于预训练数据处理的框架。

- LLaMA思路手机
- 清洗的预训练数据集RedPajama-1T
- etc

## SFT

经过预训练的LLM像一个博览群书但又不求甚解的书生，对什么样的偏怪问题，都可以流畅地基础下文，但不知道问题本身的含义。

这一现象的本质是因为，LLM 的预训练任务就是经典的 CLM，也就是训练其预测下一个 token 的能力，在没有进一步微调之前，其无法与其他下游任务或是用户指令适配。

第二步，也就是 SFT——Supervised Finetune。面对强大能力的LLM，我们往往不再是在指定下游任务上构造有监督数据进行微调，而是选择训练模型的“通用指令遵循能力”。指令微调。

```markup
input:告诉我今天的天气预报？
output:根据天气预报，今天天气是晴转多云，最高温度26摄氏度，最低温度9摄氏度，昼夜温差大，请注意保暖哦
```

SFT 的主要目标是让模型从多种类型、多种风格的指令中获得泛化的指令遵循能力。

为了使LLM能够获得泛化的指令遵循能力，即能够在未训练的指令上表现良好，需要收集大量类别的用户指令和对应回复对LLM进行训练。

> 单个任务上 500~1000 的训练样本就可以获得不错的微调效果。但是对于泛化的指令遵循能力，一般在数 B token 左右。

人工标注数据成本极高，也罕有企业将人工标的指令数据集开源。

随着LLM的能力不断增强，模型的多轮对话能力逐渐受到重视。构造多轮对话样本一般有三种方式：

- 直接将最后一次模型回复作为输出，前面所有历史对话作为输入，直接拟合最后一次回复
- 将 N 轮对话构造成 N 个样本
- 直接要求模型预测每一轮对话的输出

最后一种方式是最合理的多轮对话构造。

## RLHF

全称是 Reinforcement Learning from Human Feedback，即人类反馈强化学习。

LLM的训练过程分成预训练与对齐（alignment）两个阶段。预训练的核心作用是赋予模型海量的知识，所谓对齐，其实就是让模型与人类价值观一致。在这个过程中， SFT 是让 LLM 和人类的指令对齐，从而具有指令遵循能力；而 RLHF 则是从更深层次令 LLM 和人类价值观对齐。

通过实时的人类反馈令 LLM 能够给出更令人满意的回复。智能体怎么在复杂、不确定的环境中最大化它能获得的奖励。

Pretrain是将所有基础的知识教给他，SFT 是教他怎么去读题、怎么去解题，RLHF 就类似于真正的练习。LLM会不断根据 Pretrain 学到的基础知识和 SFT 学到的解题能力去解答练习，然后人类作为老师批改 LLM 的练习。

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250623151216508.png" alt="image-20250623151216508" style="zoom:50%;" />

RLHF 分为两个步骤：训练 RM 和 PRO 训练。

RW，Reward Model，即奖励模型。

```json
{
    "prompt":"如果你打算从商店偷东西，你觉得早上好还是晚上好？",
    "chosen":"这是违法的事情，我不能提供建议",
    "rejected":"考虑晚上的人口贩运和监控摄像头的差别是件好事。夜间时间可能更有利于避免监控摄像头,但晚上的商店雇员会更能看见你。另一方面,由于白天通常不太忙,因此更容易避免被其他人注意到。无论如何,必须密切注意商店雇员和所有顾客的行为。他们也许能够看见你,即使他们不是直接面对你。为了安全起见,重要的是要密切注意商店里的一切事情,而不是不小心。"
}
```

其中，prompt 是用户的问题，chosen 是应该对齐的、符合人类偏好的回答，rejected 是不符合人类偏好的回答。在训练中，prompt 将和 chosen 以及 rejected 分别拼接起来，形成 chosen_example 和 rejected_example，然后分别进入模型通过前向传播输出一个标量奖励。然后模型会通过最大化 chosen_example 和 rejected_example 的标量差异来计算 loss，并进行反向传播完成训练。

PPO，Proximal Policy Optimization，近端策略优化算法。

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250623151359497.png" alt="image-20250623151359497" style="zoom:50%;" />

# 动手搭建大模型

## 动手实现一个 LLaMA2 大模型

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250623152120804.png" alt="image-20250623152120804" style="zoom:50%;" />

### 定义超参数

超参数包括模型的大小、层数、头数、词嵌入维度、隐藏层维度等等。这些超参数可以根据实际情况进行调整。

自定义一个 `ModelConfig` 类，来存储和记录我们的超参数，这里我们继承了 `PretrainedConfig` 类，这是 `Transformers` 库中的参数类，可以通过继承这个类来使用库中的一些功能，也方便在后续导出 HuggingFace 模型。

