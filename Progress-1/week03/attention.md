# Transformer架构

## 注意力机制

### 神经网络

1. 基础的神经网络为MLP（多层感知机）

   其结构就是输入层 -----> 隐藏层 ----> 输出层

   每层之间使用全连接（Fully Connected）层 + 激活函数（relu）

   每层神经元与上下层的每一个神经元是完全连接的

   > 全连接设计初衷：在对输入没有任何**先验结构知识**的前提下，通过让每层神经元都连接所有输入，使模型具备**最大的信息流动能力与表达能力**，从而能逼近任何复杂函数。

   > 线性层是“搬运工”，只能对信息加权合并。
   >
   > 激活函数是“加工机器”，赋予信息复杂的形态变化能力。
   >
   > 没有激活函数，所有搬运在多次，还是没加工。

2. 为什么采用全连接？

   - 通用性：没有结构假设时的默认选择

     在图像、序列、图结构等任务中，我们有空间，时间等结构可以利用。

     在最通用的情况，比如表格数据、抽象特征输入时，我们对输入之间没有空间顺序/局部结构一无所知。

     那就只能让每个神经元都试试看所有输入，避免信息丢失。

   - 理论支持：通用逼近定理

     数学上有个经典结论，**任意连续函数都可以用一层隐藏层的全连接神经网络逼近**，前提是激活函数非线形且宽度足够。

3. 什么是激活函数？

   - 打破线性限制

     没有激活函数的神经网络本质是多个线性层的组合仍然是一个线性变换。

     这样的网络**再深也学不会复杂的决策边界**，只能你和线性可分的任务。

     引入激活函数，则变为深度学习，**每层都在非线形地提取抽象特征**。

   - 帮助神经网络拟合复杂的函数

     比如图像分类，语音识别、自然语言处理等任务，输入和输出之间都是高度非线形关系。

     激活函数可以让网络学习 ”非线形映射“，逐层提取越来越抽象的特征。

   - 引入非线形决策边界

     对于分类任务，比如将点分为两个类别，线性模型只能**画一条直线**来分。

     引入激活函数后，神经网络可以学习出**弯曲、复杂的边界**。

### 前馈神经网络

Feedforward Neural Network，FNN/FFN

MLP是前馈神经网络最典型、最基本的一种。

           ┌────────────┐
           │ 前馈神经网络 │  ← 大概念（Feedforward NN）
           └────┬───────┘
                │
      ┌─────────▼──────────┐
      │   多层感知机 MLP     │  ← 最常见的前馈结构
      └─────────┬──────────┘
                │
      ┌─────────▼─────────┐
      │  基础神经网络（常指它）│  ← 教科书里的初学模型
      └───────────────────┘



### 注意力机制

即，我们往往无需看清楚全部内容，而仅将注意力集中在重点部分即可。

具体而言，注意力机制的特点是通过计算 **Query**与**Key**的相关性为真值加权求和，从而拟合序列中每个词同其他词的相关关系。

#### 深入理解注意力机制

三个核心变量 ：查询值 Query， 键值 Key 和 真值 Value。

```json
{
    "apple":10,
    "banana":5,
    "chair":2
}
```

我们如果想要匹配的Query是一个包含多个Key的概念？例如，“fruit”，此时，我们应该将apple和banana都匹配到，但不能匹配到chair。因此，我们往往会将Key对应的Value进行组合得到最终的Value。

权重赋值如下。

```json
{
    "apple":0.6,
    "banana":0.4,
    "chair":0
}
```

我们最终查询到的值应该是：
$$
value = 0.6 * 10 + 0.4 * 5 + 0 * 2 = 8
$$
直观上，Key与Query相关性越高，则其所赋予的注意力权重就越大。

向量之间，使用点积来衡量相似性。语义相似，点击应该大于0，而语义不相似应该小于0。
$$
x = qK^T
$$
可以通过一个 Softmax 层将其转换为1的权重。

注意力机制的基本公式：
$$
attention(Q,K,V) = softmax(qK^T)v
$$
此时的值还是一个标量，同时，我们此次只查询了一个Query。我们可以将值转换为维度为 d~v~ ，同时一次查询多个Query，同样将多个Query对应的词向量堆叠在一起形成一个矩阵Q，得到：
$$
attention(Q,K,V) = softmax(QK^T)V
$$
如果**Q和K对应的维度d~k~ 比较大**，softmax 缩放时就非常容易影响，使不同值之间的差异较大，从而影响梯度的稳定性。因此，我们要将Q和K乘积的结果做一个放缩：
$$
attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d~k~}}V)
$$

> 当d~k~ 很大时，QK^T^ 的值会非常大，造成：
>
> - softmax的指数部分爆炸（例如 e^50^ =5.185×10²¹)
> - 结果接近 one-hot 分布，梯度消失或爆炸
> - 训练过程变得不稳定
>
> 这个技巧类似于BatchNorm，LayerNorm的动机：控制数值分布，帮助训练稳定。

#### 自注意力

实际应用中，我们往往只需要计算Query和Key之间的注意力结果，很少存在额外的真值Value。在Transfomer的Decoder结构中，Q来自于Decoder的输入，K和V来自于Encoder的输出，从而拟合了编码信息与历史信息之间的关系。

在Transformer的Encoder结构中，使用的是注意力机制的变种，自注意力（self- attention）。即是计算本身序列中**每个元素对其他元素的注意力分布**。Q、K、V都由同一个输入通过不同的参数矩阵计算得到。在Encoder中，Q、K、V分别是输入对参数矩阵 W~q~, W~k~, W~v~，做积得到，从而拟合输入语句中每一个token对其他所有token的关系。

通过自注意力机制，我们可以找一段文本中每个token与其他所有token的相关关系大小，从而建模文本之间的依赖关系。在代码中的实现，self-attention 机制其实是通过给 Q、K、V 的输入传入同一个参数实现的：

```python
# attention 为上文定义的注意力计算函数
attention(x, x, x)
```

#### 掩码自注意力

Mask Self-Attention，使用注意力掩码的自注意力机制。作用是遮蔽一些特定位置的token，模型在学习的过程中，会忽略掉被遮蔽的token。

使用注意力机制的Transfomer模型，也是通过类似于 n-gram 的语言模型任务来学习的，也就是对一个文本序列，不断根据之间的token来预测下一个token，直到将整个文本序列补全。

n-gram：

```markup
Step 1：输入 【BOS】，输出 I
Step 2：输入 【BOS】I，输出 like
Step 3：输入 【BOS】I like，输出 you
Step 4：输入 【BOS】I like you，输出 【EOS】
```

transformer：

```markup
<BOS> 【MASK】【MASK】【MASK】【MASK】
<BOS>    I   【MASK】 【MASK】【MASK】
<BOS>    I     like  【MASK】【MASK】
<BOS>    I     like    you  【MASK】
<BoS>    I     like    you   </EOS>
```

在每一行输入中，模型只看到前面的token，预测下一个token。上述过程不再是串行的过程，而是可以一起并行地输入到模型中。模型只需要每一个样本根据未被遮蔽的token来预测下一个token即可，从而实现了并行的语言模型。

#### 多头注意力

一次注意力计算只能拟合一种相关关系，单一的注意力机制很难全面拟合语句序列里的相关关系。因此Transformer使用了多头注意力机制，即同时对一个语料进行多次注意力计算，每次注意力计算都能拟合不同的关系，将最后的多次结果拼接起来作为最后的输出。

将原始的输入序列进行多组的自注意力处理；然后再将每一组得到的自注意力结果拼接起来，再通过一个线形层进行处理，得到最终的输出：
$$
MultiHead(Q,K,V)=Concat(head_1,head_2,...,head_h)W^O   
\newline
where\ head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)
$$

```python
import torch
# 模拟两个注意力头的输出，每个头形状为 (1, 3, 4)
head1 = torch.tensor([[[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]]], dtype=torch.float32)

head2 = torch.tensor([[[10, 20, 30, 40],
                       [50, 60, 70, 80],
                       [90, 100, 110, 120]]], dtype=torch.float32)
                       
# dim=-1 表示沿着最后一个维度拼接（即 feature 维度）
output = torch.cat([head1, head2], dim=-1)

print(output.shape)  # (1, 3, 8)
print(output)

tensor([[[  1.,   2.,   3.,   4.,  10.,  20.,  30.,  40.],
         [  5.,   6.,   7.,   8.,  50.,  60.,  70.,  80.],
         [  9.,  10.,  11.,  12.,  90., 100., 110., 120.]]])
```

```python
import torch.nn as nn
import torch

"""多头注意力计算模块"""
class MultiHeadAttention(nn.Module):
  
  def __init__(self, args: ModelArgs, is_causal=False):
    # 构造函数
    # args：配置对象
    super().__init__()
    # 隐藏层维度必须是头数的整数倍，因为后面我们将会输入拆成头数个矩阵
    assert args.n_embd % args.n_heads == 0
    # 模型并行处理大小，默认为1。
    model_parallel_size = 1
    # 本地计算头数，等于总头数除以模型并行处理大小。
    self.n_local_heads = args.n_heads // model_parallel_size
    # 每个头的维度，等于模型维度除以头的总数
    self.head_dim = args.dim // args.n_heads
    
    # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x n_embd
    # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
    # 不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
     self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
     self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
     self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
     # 输出权重矩阵，维度为 n_embd x n_embd（head_dim = n_embeds / n_heads）
     self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
     # 注意力的 dropout
     self.attn_dropout = nn.Dropout(args.dropout)
     # 残差连接的 dropout
     self.resid_dropout = nn.Dropout(args.dropout)
     
  def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):
    
    # 获取批次大小和序列长度，[batch_size, seq_len, dim]
    bsz, seqlen, _ = q.shape
    
    # 计算查询 Q,K,V
		xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)
    
    # 将 Q,K,V 拆分成多头
    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2) 
    
    # 注意力计算
    scores = torch.matmul(xq, xk.transpose(2,3)) / math.sqrt(self.head_dim)
    # 计算softmax
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    # 做dropout
    scores = self.attn_dropout(scores)
    # ✖️v
    output = torch.matmul(scores, xv)
    
    # 拼接
   	output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

    # 最终投影回残差流。
    output = self.wo(output)
    output = self.resid_dropout(output)
    return output
    
```



## Encoder-Decoder

### Seq2Seq模型

模型输入是一个自然语言序列，输出是一个可能不等长的自然语言序列。

### 实现前馈神经网络

```python
class MLP(nn.Module):
  
  def __init__(self, dim:int, hidden_dim:int, dropout:float):
    super().__init__()
    # 定义第一层线性变换，从输入维度到隐藏维度
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    # 定义第二层线性变换，从隐藏维度到输入维度
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    # 定义dropout层，用于防止过拟合
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    # 前向传播函数
    # 首先，输入x通过第一层线性变换和RELU激活函数
    # 然后，结果乘以输入x通过第三层线性变换的结果
    # 最后，通过第二层线性变换和dropout层
    return self.dropout(self.w2(F.relu(self.w1(x))))
```

Transformer的前馈神经网络是由两个线形层中间加一个 RELU 激活函数组成，以及前馈神经网络还加入了一个 Dropout 层来防止过拟合。

### 层归一化

Layer Norm

归一化的核心是为了让不同层输入的取值范围或者分布能够比较一致。

由于深度神经网络中每一层的输入都是上一次的输出，多层传递，之前所有的神经层的参数变化会导致其输入的分布发生较大的改变。

在深度神经网络中，需要归一化操作，将每一层的输入都归一化成标准正态分布。

归一化存在缺陷：

- 当显存有限，mini-batch 较小时，Batch Norm 取的样本的均值和方差不能反映全局的统计分布信息，从而导致效果变差；
- 对于在时间维度展开的 RNN，不同句子的同一分布大概率不同，所以 Batch Norm 的归一化会失去意义；
- 在训练时，Batch Norm 需要保存每个 step 的统计信息（均值和方差）。在测试时，由于变长句子的特性，测试集可能出现比训练集更长的句子，所以对于后面位置的 step，是没有训练的统计量使用的；
- 应用 Batch Norm，每个 step 都需要去保存和计算 batch 统计量，耗时又耗力

因此使用效果更好的层归一化。Layer Norm 在每个样本上计算其所有层的均值和方差，从而使每个样本的分布达到稳定。

简单实现一个 Layer Norm：

```python
class LayerNorm(nn.Module):
  def __init__(self, features, eps=1e-6):
    super(LayerNorm, self).__init__()
    # 线性矩阵做映射
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps
    
  def forward(self, x):
    # 在统计每个样本所有维度的值，求均值和方差
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    
    # 注意这里也在最后一个维度发生了广播
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

### 残差连接

Transformer模型结构复杂，层数深，为了避免模型退化，采用了残差连接的思想。

### Encoder

实现单个Encoder Layer：

```python
class EncoderLayer(nn.Module):
  def __init__(self, args):
    super().__init__()
    # 一个 Layer 有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
    self.attention_norm = LayerNorm(args.n_embd)
    # Encoder不需要掩码
    self.attention = MultiHeadAttention(args, is_causal=False)
    self.fnn_norm = LayerNorm(args.n_embd)
    self.feed_forward = MLP(args)
    
  def forward(self, x):
    # Layer Norm
    norm_x = self.attention_norm(x)
    # 自注意力
    h = x + self.attention.forward(norm_x,norm_x,norm_x)
    # 通过前馈神经网络
    out = h + self.feed_forward.forward(self.fnn_norm(h))
    return out
```

### Decoder

实现单个Decoder Layer：

```python
class DecoderLayer(nn.Module):
  def __init__(self,args):
    super().__init__()
    # 一个 Layer 中有三个 LayerNorm， 分别在 Mask Attention之前，Self-Attention 之前和 MLP 之前
    self.attention_norm_1 = LayerNorm(args.n_embd)
    # Decoder 的第一个部分是 Mask attention
    self.mask_attention = MultiHeadAttention(args, is_causal=True)
    self.attention_norm_2 = LayerNorm(args.n_embd)
    # Decoder 的第二个部分 类似于 Encoder 的 Attention
    self.attention = MultiHeadAttention(args, is_causal=False)
    self.ffn_norm = LayerNorm(args.n_embd)
    # 第三个部分是 MLP
    self.feed_forward = MLP(args)
    
  def forward(self, x , enc_out):
    # Layer Norm
    norm_x = self.attention_norm_1(x)
    # 掩码自注意力
    x = x + self.mask_attention.forward(norm_x,norm_x,norm_x)
    # 多头注意力
    norm_x = self.attention_norm_2(x)
    h = x + self.attention.forward(norm_x, enc_out, enc_out)
    # 经过前馈神经网络
    out = h + self.feed_forward.forward(self.fnn_norm(h))
		return out
```

## 搭建一个Transformer

### Embedding层

自然语言转换为机器可以处理的向量。

分词器把自然语言切分成token并转换成一个固定的index。

Embedding层的输入往往是一个形状为 (batch_size, seq_len, 1) 的矩阵。

- 第一个维度是一次批处理的数量

- 第二个维度是自然语言序列的长度

- 第三个维度是token经过tokenizer转换成的index的值

- i.e : "我喜欢你" -> [[0,1,2]]，其 batch_size为1，seq_len为3，转换的index如下

  ```
  input: 我
  output: 0
  
  input: 喜欢
  output: 1
  
  input：你
  output: 2
  ```

Embedding内部是一个可训练的（vocab_size, embedding_dim)的权重矩阵，词表里的每一个值，都对应一行维度为 embedding_dim 的向量。对于输入的值，会对应到这个词向量，然后拼接成 **(batch_size, seq_len, embedding_dim)** 的矩阵输出。

可以直接使用 torch 中的 Embedding 层：

```python
self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
```

### 位置编码

注意力机制中序列的每一个 token，对其来说都是平等的，即”我喜欢你“ 和 ”你喜欢我“ 在注意力机制看来是完全相同的。因此使用了位置编码。

Transfomer使用了正余弦函数来进行位置编码（绝对位置编码Sinusoidal），其编码方式为：
$$
PE(pos,2i) = sin(pos/10000^{2i/d_{model}})
\newline
PE(pos,2i+1) = cos(pos/10000^{2i/d_{model}})
$$
举例，输入为长度为4的句子“I like to code”，我们可以得到下面的词向量矩阵 x，其中每一行代表的就是一个词向量，x~0~ = [0.1, 0.2, 0.3, 0.4] 对应的就是 “I” 的词向量，他的 pos 就是0，以此类推。

![截屏2025-06-17 13.43.06](/Users/edward_beck8n24/Desktop/截屏2025-06-17 13.43.06.png)

位置编码的好处：

1. 使 PE 能够适应比训练集里面所有句子更长的句子，假设训练集里面最长的句子是有20个单词，突然来了一个长度为21的句子，使用公式，可以计算出第21位的 Embedding。
2. 可以让模型容易地计算出相对位置，对于固定长度的间距 k，PE(pos+k)，可以用 PE(pos) 计算得到。

```python
def PositionalEncoding(nn.Module):
  """位置编码模块"""
  
  def __init__(self, args):
    super(PositionalEncoding, self).__init__()
    # dropout 层
    self.dropout = nn.Dropout(p=args.dropout)
    
    # block size 是序列的最大长度
    pe = torch.zeros(args.block_size, args.n_embd)
    position = torch.arange(0, args.block_size).unsqueeze(1)
    
    # 计算theta
    div_term = torch.exp(
      torch.arange(0, args.n_embd, 2) * (math.log(10000) / args.n_embd)
    )
    
    # 分别计算 sin、cos的结果
    pe[:,0::2] = torch.sin(position * div_term)
    pe[:,1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer("pe", pe)
    
  def forward(self, x):
    # 将位置编码加到 Embedding 结果上
    x = x + self.pe[:, :x.size(1)].requires_grad_(False)
    return self.dropout(x)
  
```

### 一个完整的Transformer

```python
class Transformer(nn.Module):
  """整体模型"""
  def __init__(self,args):
    super().__init()
    # 必须输入词表大小和 block size
    assert args.vocab_size is not None
    assert args.block_size is not None
    self.args = args
    self.transformer = nn.ModuleDict(dict(
    	wte = nn.Embedding(args.vocab_size, args.n_embd),
      wpe = PositionalEncoding(args),
      drop = nn.Dropout(args.dropout),
      encoder = Encoder(args)
      decoder = Decoder(args)
    ))
    
    # 最后的线形层， 输入时 n_embd，输出时词表大小
    self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
    
    # 初始化所有的权重
    self.apply(self._init_weights)
    
    # 查看所有的参数的数量
    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
   
  """前向计算函数"""
	def forward(self, idx, targets=None):
    # 输入为 idx， 维度为（batch size, sequence length), targets计算loss
    device = idx.device
    b, t = idx.size()
    assert t <= self.args.block_size f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"
    
    # 首先通过 self.transformer
    # 首先将输入 idx 通过 Embedding层，得到维度为 （batch size, sequence length, n_embd)
    # 通过Embedding层
    tok_emb = self.transformer.wte(idx)
    # 通过位置编码
    pos_emb = self.transformer.wpe(tok_emb)
    # 再进行 Dropout
    x = self.transformer.drop(pos_emb)
    # 然后通过 Encoder
    enc_out = self.transformer.encoder(x)
    # 再通过 Decoder
    x = self.transformer.decoder(x,enc_out)
    
    if targets is not None:
      logits = self.lm_head(x)
      loss = F.cross_entropy(logits.view(-1,logits.size(-1), targets.view(-1), ignore_index=-1))
    else:
      logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
     	loss = None

    return logits, loss
```

