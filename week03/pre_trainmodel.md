# 预训练语言模型

## Encoder-only PLM

针对 Encoder、Decoder 的特点，引入 **ELMo** 的预训练思路，开始出现不同的、对 Transformer 进行优化的思路。

### BERT

Bidirectional Encoder Representations from Transformers.

自BERT推出以来，**预训练+微调**的模式开始成为自然语言处理任务的主流。

#### (1) 思想沿承

- Transformer, BERT在Transformer的模型基座上进行优化，通过将Encoder结构进行堆叠，扩大模型参数，打造了在NLU任务上(**Natural Language Understanding**)独居天分的模型架构。
- 预训练+微调范式。引入更适合文本理解、能捕捉深层双向语义关系的预训练任务MLM，将与训练-微调范式推向了高潮。(**MLM,Masked Language Modeling**)

#### (2) 模型架构——Encoder Only

BERT是针对NLU任务打造的预训练模型，其输入一般是文本序列，而输出一般是Label，例如情感分类的积极、消极Label。

使用Encoder堆叠而成的BERT本质上也是一个Seq2Seq模型，只是**没有加入对特定任务的Decoder**，因此，为适配各种NLU任务，在模型的最顶层加入了一个分类头 prediction_heads，用于将多维度的隐藏状态通过线形层转换到分类维度。

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250620144707892.png" alt="image-20250620144707892" style="zoom:50%;" />

输入的文本序列首先通过 tokenizer(分词器)，转化成 input_ids，然后进入 Embedding 层转换为特定维度的 hidden_states，再经过 Encoder 块。Encoder 块中是对叠起来的 N 层 Encoder Layer，BERT 有两种规律的模型，分别是 base 版本（12层 Encoder Layer， 768 的隐藏层维度，总参数量110M），large版本（24层 Encoder Layer，1024 的隐藏层维度，总参数量340M）。

通过Encoder编码之后的最顶层 hidden_states 最后经过 prediction_heads 就得到了最后的类别概率，经过 Softmax 计算就可以计算出模型预测的类别。

Prediction_heads其实就是线形层加上激活函数，一般而言，最后一个线性层的输出维度和任务的类别数相等：

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250620145729586.png" alt="image-20250620145729586" style="zoom:50%;" />

而每一层 Encoder Layer 都是和 Transformer 中的 Encoder Layer 结构类似的层：

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250620145818102.png" alt="image-20250620145818102" style="zoom:50%;" />

已经通过 Embedding 层映射的 hidden_states 进入核心的 attention 机制，然后通过残差连接的机制和原输入相加，再经过一层 Intermediate 层得到最终输出。Intermediate 层是 BERT 的特殊称呼，其实就是一个线性层加上激活函数：

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250620145920367.png" alt="image-20250620145920367" style="zoom:50%;" />

BERT所使用的激活函数是 GELU 函数，高斯误差线性单元激活函数。
$$
GELU(x)=0.5x(1+tanh(\sqrt{
π/
2}
 )(x+0.044715x^
3
 )
$$
GELU函数的核心思路为将随机正则的思想引入激活函数，通过输入自身的概率分布，来决定抛弃还是保留自身的神经元。

BERT的注意力机制和 Transformer 中 Encoder 的自注意力机制几乎完全一致，但是 BERT 将相对位置编码融合在了注意力机制中，将相对位置编码同样视为可训练的权重参数。

与Transformer唯一的差异在于，在完成注意力分数的计算之后，先通过 Positional Embedding 层来融入相对位置信息。这里的 Positional Embedding，其实就是一层线性矩阵。通过可训练的参数来拟合相对位置，相比Transformer使用的绝对位置的编码，Sinusoidal 能够**拟合更丰富的相对位置**信息。但是这样也**增加了不少模型参数**，同时完全无法处理超过模型训练长度的输入（例如，对BERT而言，能处理的最大上下文长度是512个token。）

#### (3) 预训练任务 —— MLM + NSP（next sentence prediction)

BERT更大的创新点在于提出两个新的预训练任务上——MLM，NSP。

预训练-微调范式的核心优势在于，通过将预训练和微调分离，完整一次预训练的模型可以仅通过微调应用在几乎所有下游任务上。只要微调的成本较低，即使预训练成本是之前的数十倍，模型仍然有更大的应用价值。

预训练数据的核心要求是需要极大的数据规模（数亿token）。预训练的数据一定是从无监督的语料中获取的。因此，互联网上所有的文本语料都可以被用于预训练。

LM预训练任务的一大缺陷在于，其直接拟合从左到右的语义关系，但忽略了双向语义关系。因此，有没有一种预训练任务，能够既利用海量无监督语料，又能够训练模型拟合双向语义关系？

MLM于是被提出。掩码语言模型作为新的预训练任务。MLM模拟的是“完形填空”。在一个文本序列中随机遮蔽部分 token，然后将所有未被遮蔽的 token 输出模型，要求模型根据输入预测被遮蔽的 token。例如：

```markup
输入：I <MASK> you because you are <MASK>
输出：<MASK> - love; <MASK> - wonderful
```

MLM任务无需对文本进行任何人为的标注，只需要对文本进行随机遮蔽即可，因此也可以利用互联网所有文本语料实现预训练。

#### (4) 下游任务微调

在海量无监督语料上预训练来获得通用的文本理解与生成能力，再在对应的下游任务上进行微调。这种思想的一个重点在于，预训练得到的强大能力能否通过低成本的微调快速迁移到对应的下游任务中。

在完成预训练后，针对每一个下游任务，只需要使用一定量的全监督人工标注数据，对预训练的BERT在该任务上进行**微调**即可。微调，在特定的任务、更少的训练数据、更小的 batch_size 上进行训练，更新参数的幅度更小。

### RoBERTa

13GB的预训练数据是否让 BERT 达到了充分的拟合？如果我们使用更多的预训练语料，是否可以进一步增强模型性能？更多的，BERT 所选用的预训练任务、训练超参数是否是最优的？RoBERTa 应运而生。

#### (1) 优化一：去掉 NSP 预训练任务

```markup
1. 段落构建的 MLM + NSP：BERT 原始预训练任务，输入是一对片段，每个片段包括多个句子，来构造 NSP 任务；
2. 文档对构建的 MLM + NSP：一个输入构建一对句子，通过增大 batch 来和原始输入达到 token 等同；
3. 跨越文档的 MLM：去掉 NSP 任务，一个输入为从一个或多个文档中连续采样的完整句子，为使输入达到最大长度（512），可能一个输入会包括多个文档；
4. 单文档的 MLM：去掉 NSP 任务，且限制一个输入只能从一个文档中采样，同样通过增大 batch 来和原始输入达到 token 等同
```

RoBERTa在预训练中去掉了 NSP，只使用 MLM 任务。

同时，RoBERTa 对 MLM 任务本身也做出了改进。在 BERT 中，Mask 的操作是在数据处理阶段完成的，因此后期预训练时同一个 sample 待预测的 <mask> 总是一致的。而 RoBERTa 将 Mask 操作放到了训练阶段，也就是动态遮蔽策略，从而让每一个 Epoch 的训练数据 Mask 的位置都不一致。在实验中，动态遮蔽仅有很微弱的优势优于静态遮蔽，但由于动态遮蔽更高效、易于实现，后续 MLM 任务基本都使用了动态遮蔽。

#### (2)  优化二：更大规模的预训练数据和预训练步长

RoBERTa 使用了更大量的无监督语料进行预训练。

RoBERTa 认为更大的 batch size 既可以提高优化速度，也可以提高任务结束性能。

#### (3)  优化三：更大的 bpe 词表

RoBERTa、BERT 和 Transformer 一样，都使用了 BPE 作为 Tokenizer 的编码策略。BPE，即 Byte Pair Encoding，字节对编码，是指以子词对作为分词的单位。例如，对“Hello World”这句话，可能会切分为“Hel，lo，Wor，ld”四个子词对。而对于以字为基本单位的中文，一般会按照 字节编码进行切分。例如，在 UTF-8 编码中，“我”会被编码为“E68891”，那么在 BPE 中可能就会切分成“E68”，“891”两个字词对。

BERT 原始的 BPE 词表大小为 30K，RoBERTa 选择了 50K 大小的词表来优化模型的编码能力。

通过上述三个部分的优化，RoBERTa 成功地在 BERT 架构的基础上刷新了多个下游任务的 SOTA，也一度成为 BERT 系模型最热门的预训练模型。同时，RoBERTa 的成功也证明了**更大的预训练数据、更大的预训练步长的重要意义，这也是 LLM 诞生的基础之一。**

### ALBERT

ALBERT 成功地以更小规模的参数实现了超越 BERT 的能力。

#### (1) 优化一：将 Embedding 参数进行分解

而从另一个角度看，Embedding 层输出的向量是我们对文本 token 的稠密向量表示，从 Word2Vec 的成功经验来看，这种词向量并不需要很大的维度，Word2Vec 仅使用了 100维大小就取得了很好的效果。因此，Embedding 层的输出也许不需要和隐藏层大小一致。

ALBERT 对 Embedding 层的参数矩阵进行了分解，让 Embedding 层的输出维度和隐藏层维度解绑，也就是在 Embedding 层的后面加入一个线性矩阵进行维度变换。

#### (2) 优化二：跨层进行参数共享

由于 24个 Encoder 层带来了巨大的模型参数，因此，ALBERT 提出，可以让各个 Encoder 层共享模型参数，来减少模型的参数量。

上述优化虽然极大程度减小了模型参数量并且还提高了模型效果，却也存在着明显的不足。虽然 ALBERT 的参数量远小于 BERT，但训练效率却只略微优于 BERT，因为在模型的设置中，虽然各层共享权重，但计算时仍然要通过 24次 Encoder Layer 的计算，也就是说训练和推理时的速度相较 BERT 还会更慢。这也是 ALBERT 最终没能取代 BERT 的一个重要原因。

#### (3) 优化三：提出 SOP 预训练任务

不同于 RoBERTa 选择直接去掉 NSP，ALBERT 选择改进 NSP，增加其难度，来优化模型的预训练。

```markup
输入：
    Sentence A：I love you.
    Sentence B: Because you are wonderful.
输出：
    1（正样本）

输入：
    Sentence A：Because you are wonderful.
    Sentence B: I love you.
输出：
    0（负样本）
```

ALBERT 通过实验证明，SOP 预训练任务对模型效果有显著提升。使用 MLM + SOP 预训练的模型效果优于仅使用 MLM 预训练的模型更优于使用 MLM + NSP 预训练的模型。

## Encoder-Decoder PLM

BERT也存在一些问题，例如MLM任务和下游任务微调的不一致性，以及无法处理超过模型训练长度的输入等问题。

### T5

T5（Text-to-Text Transfer Transformer）是由 Google 提出的一种预训练语言模型，通过将所有 NLP 任务统一表示为文本到文本的转换问题，大大简化了模型设计和任务处理。

#### (1) 模型结构：Encoder-Decoder

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250621084912970.png" alt="image-20250621084912970" style="zoom:50%;" />

T5主要包括 Self-attention 和 前馈神经网络。Self-attention 用于捕捉输入**序列中的全局依赖关系**，前馈神经网络用于**处理特征的非线性变化**。

与原始的 Transformer 模型不同， T5 的模型 LayerNorm 采用了 RMSNorm，通过计算每个神经元的均方根（Root Mean Square）来归一化每个隐藏层的激活值。RMSNorm 的参数设置与 Layer Normalization 相比更简单，只有一个可调参数，可以更好地适应不同的任务和数据集。
$$
RMSNorm(x) = \frac{x}{\sqrt{1/n\sum_{i=1}^n  w_i + \epsilon}}
$$

> (x) 是层的输入。
>
> (w~i~) 是层的权重。
>
> (n) 是权重的数量。
>
> (*ϵ*) 是一个小常数，用于数值稳定性（以避免除以零的情况）。

这种归一化有助于通过确保权重的规模不会变得过大或过小来稳定学习过程，这在具有许多层的深度学习模型中特别有用。

#### (2) 预训练任务

训练所使用的数据集是一个大规模的文本数据集，包含了各种各样的文本数据，如维基百科、新闻等等。对数据经过细致的处理后，生成了用于训练的750GB的数据集 C4。

- 多任务预训练：T5 还尝试了将多个任务混合在一起进行预训练，不仅仅是单独的MLM任务。这有助于模型学习更通用的语言表示。
- 预训练到微调的转换：预训练完成后，T5模型会在下游任务上进行微调。微调时，模型在任务特定的数据集上进行训练，并根据任务调整解码策略。

通过大规模预训练，T5模型在多个NLP任务上取得了优异的性能，预训练时T5成功的关键因素之一。

#### (3) 大统一思想

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250621090540362.png" alt="image-20250621090540362" style="zoom:50%;" />

对于不同的NLP任务，每次输入前都会加上一个任务描述前缀，明确指定当前任务的类型。这不仅帮助模型在预训练阶段学习到不同任务之间的通用特证，也便于在微调阶段迅速适应具体任务。

例如，任务前缀可以是“summarize: ”用于摘要任务，或“translate English to German: ”用于翻译任务。

T5的大一统思想通过将所有NLP任务统一为文本到文本的形式，简化了任务处理流程，增强了模型的通用性和适应性。这一思想不仅推动了自然语言处理技术的发展，也为实际应用提供了更为便捷和高效的解决方案。

## Decoder-Only PLM

### GPT

首先明确提出预训练-微调思想的模型其实是 GPT。

#### (1) 模型架构 —— Decoder Only

Decoder 层仅保留了一个带掩码的注意力层，并且将 LayerNorm 层从 Transformer 的注意力层之后提到了**注意力层之前**。

另一个结构上的区别在于，GPT的MLP层没有选择线性矩阵来进行特征提取，而是选择了两个一维卷积层来提取，不过从效果上来说，这两者没有太大区别。

#### (2) 预训练任务—— CLM

Decoder-Only 的模型结构往往更适合于文本生成任务，因此，Decoder-Only模型往往选择了最传统也最直接的预训练任务——因果语言模型，Casual Language Model。

CLM基于一个自然语言序列的前面所有 token 来预测下一个 token，通过不断重复该过程来实现目标文本序列的生成。CLM 是一个经典的补全形式。

```markup
input: 今天天气
output: 今天天气很

input: 今天天气很
output：今天天气很好
```

很明显，CLM是更直接的预训练任务，其天生和人类书写自然语言文本的习惯相契合，也和下游任务直接匹配，相对于MLM任务更加直接。

#### (3) GPT 的发展

zero-shot（零样本学习）

在大模型时代，zero-shot 及其延伸出的 few-shot（少样本学习）才开始逐渐成为主流。

之所以说 GPT-3 是 LLM 的开创之作，除去其巨大的体量带来了涌现能力的凸显外，还在于其提出了 few-shot 的重要思想。

### LLaMA

#### (1) 模型架构—— Decoder Only

在 decoder 层中，hidden_states 会经历一系列的处理，这些处理由多个 decoder block 组成。每个 decoder block 都是模型的核心组成部分。他们负责对 hidden_states 进行深入的分析和转换。



在完成masked self-attention层之后，hidden_states会进入MLP层。在这个多层感知机层中，模型通过两个全连接层对hidden_states进行进一步的特征提取。第一个全连接层将hidden_states映射到一个中间维度，然后通过激活函数进行非线性变换，增加模型的非线性能力。第二个全连接层则将特征再次映射回原始的hidden_states维度。

最后，经过多个decoder block的处理，hidden_states会通过一个线性层进行最终的映射，这个线性层的输出维度与词表维度相同。这样，模型就可以根据hidden_states生成目标序列的概率分布，进而通过采样或贪婪解码等方法，生成最终的输出序列。这一过程体现了LLaMA模型强大的序列生成能力。

### GLM

智谱开发的主流中文 LLM 之一。

#### (1) 模型架构——略微不同

- 使用 Post Norm 而非 Pre Norm。因此，对于更大体量的模型来说，一般认为 Pre Norm 效果会更好。但 GLM 论文提出，使用 Post Norm 可以避免 LLM 的数值错误（虽然主流 LLM 仍然使用了 Pre Norm）；
- 使用单个线性层实现最终 token 的预测，而不是使用 MLP；这样的结构更加简单也更加鲁棒，即减少了最终输出的参数量，将更大的参数量放在了模型本身；
- 激活函数从 ReLU 换成了 GeLUS。ReLU 是传统的激活函数，其核心计算逻辑为去除小于 0的传播，保留大于 0的传播；GeLUS 核心是对接近于 0的正向传播，做了一个非线性映射，保证了激活函数后的非线性输出，具有一定的连续性。

#### (2) 预训练任务——GLM

GLM（General Language Model，通用语言模型）任务，这也是 GLM 的名字由来。

GLM 是一种结合了自编码思想和自回归思想的预训练方法。所谓自编码思想，其实也就是 MLM 的任务学习思路，在输入文本中随机删除连续的 tokens，要求模型学习被删除的 tokens；所谓自回归思想，其实就是传统的 CLM 任务学习思路，也就是要求模型按顺序重建连续 tokens。

```markup
输入：I <MASK> because you <MASK>
输出：<MASK> - love you; <MASK> - are a wonderful person
```