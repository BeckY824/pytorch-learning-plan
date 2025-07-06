# LoRA

Low-Rank Adaptation，是一种微调大模型预训练的技术。它的核心思想是通过低秩分解减少微调时的参数量，而不牺牲模型的性能。

## 为什么需要LoRA

大模型的大，不仅体现在其参数量上，更体现在我们无法轻松进行微调。全量微调是一个预训练大模型的代价非常高，而且一般的设备根本训练不动。LoRA提供了一种高效的微调方法，使得在小型设备上微调大模型成为可能。

## LoRA的核心思想

LoRA的核心在于利用低秩分解来近似模型权重的更新。**过参数化**模型的学习特征为于一个低维的内在子空间。

### 低秩分解

在LoRA中，**模型适配或者微调过程中，权重的变化**同样是低秩的。基于这个假设，权重矩阵 w 的更新可以近似表示为两个小矩阵 B 和 A 的乘积：
$$
\Delta W = BA
$$
其中：

- A ∈ R^rxd^ , r 是低秩值 ，d 是输入特征维度
- B ∈ R^kxr^ , k 时输出特征维度

通过训练这两个小矩阵，我门可以近似低更新原始权重矩阵 w，而无需训练整个大的 w。

### 应用到神经网络中的线性层

在线性层，前向传播的计算为：
$$
y = Wx + b
$$
在微调过程中，通常需要更新 w 和 b 。但在 LoRA 中，我们可以冻结原始的 W，仅仅在其基础上添加一个可训练的赠量：$\Delta W$:
$$
y = (W + \Delta W)x + b
$$
其中：
$$
\Delta W = BA
$$
通过训练 A 和 B，我们大大减少了需要更新的参数数量。

> 不增加推理延迟。只有在推理前把 LoRA 训练得到的增量矩阵 $\Delta W$ 加回原始参数 W，才能恢复为一个普通模型结构，避免额外计算。

假设：

- 输入特征维度 d=1024
- 输出特征维度 k=1024
- 低秩值 r=4

**全量微调参数量：**

- 权重参数: 1024×1024=1,048,576
- 偏置参数: 1024
- **总参数量: 1,048,576+1024=1,049,600**

**使用 LoRA 微调参数量：**

- 矩阵 A 参数: 4×1024=4,096
- 矩阵 B 参数: 1024×4=4,096
- 偏置参数: 1024
- **总参数量: 4,096+4,096+1024=9,216**

**参数量对比：**

- 全量微调: 1,049,600 参数
- LoRA 微调: 9,216 参数
- **参数减少比例: 9,2161,049,600≈0.0088**

也就是说，使用 LoRA 后，参数量减少了约 **114 倍**，即参数量仅为原来的 **0.88**。

<img src="/Users/edward_beck8n24/Library/Application Support/typora-user-images/image-20250702153114688.png" alt="image-20250702153114688" style="zoom:50%;" />

### 代码实现：线性层的 LoRA

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
  def __init__(self, in_features, out_features, r):
    super(LoRALinear,self).__init__()
    self.in_features = in_features # 对应 d
    self.out_features = out_features # 对应 k
    self.r = r # 低秩值
    
    # 原始权重矩阵，冻结
    self.weight = nn.Parameter(torch.randn(out_features, in_features))
    self.weight.requires_grad = False # 冻结
    
    # LoRA 部分的参数，初始化 A 从均值为 0 的正态分布中采样，B 为全零
    self.A = nn.Parameter(torch.empty(r, in_features)) # 形状为 (r,d)
    self.B = nn.Parameter(torch.zeros.(out_features, r)) # 形状为 (k,r)
    nn.init.normal_(self.A, mean=0.0, std=0.02) # 初始化 A
    
    # 偏执项，可选
    self.bias = nn.Parameter(torch.zeros(out_features))
    
  def forward(self, x):
    # 原始部分
    original_output = torch.nn.functional.linear(x, self.weight, self.bias)
    # LoRA 增量部分
    delta_W = torch.matmul(self.B, self.A) # 形状为 (k,d)
    lora_output = torch.nn.functional.linear(x, delta_W)
    # 总输出
    return original_output + lora_output
    
```

## LoRA 在注意力机制中的应用

Transformer 模型的核心机制是注意力机制，其中涉及到 Query，Key，Value的计算。这些都是线性变换。

###  代码实现：带 LoRA 的注意力

```python
import torch
import torch.nn as nn

class LoRAAttention(nn.Module):
  def __init__(self, embed_dim, r):
    super(LoRAAttention, self).__init__()
    self.embed_dim = embed_dim # 对应 d_model
    self.r = r # 低秩值
    
    # 原始v的 QKV 权重，冻结
    self.W_Q = nn.Linear(embed_dim, embed_dim)
    self.W_K = nn.Linear(embed_dim, embed_dim)
    self.W_V = nn.Linear(embed_dim, embed_dim)
    self.W_O = nn.Linear(embed_dim, embed_dim)
    
    for paramedics in self.W_Q.parameters():
      param.requires_grad = False
    for param in self.W_K.parameters():
      param.requires_grad = False
    for param in self.W_V.parameters():
      param.requires_grad = False
		
    # LoRA 的 Q 部分
    self.A_Q = nn.Parameter(torch.empty(r, embed_dim))
    self.B_Q = nn.Parameter(torch.zeros(embed_dim, r))
    nn.init.normal_(self.A_Q, mean=0.0, std=0.02)
    
    # LoRA 的 K 部分
    self.A_K = nn.Parameter(torch.empty(r, embed_dim))
    self.B_K = nn.Parameter(torch.zeros(embed_dim, r))
    nn.init.normal_(self.A_K, mean=0.0, std=0.02)

    # LoRA 的 V 部分
    self.A_V = nn.Parameter(torch.empty(r, embed_dim))
    self.B_V = nn.Parameter(torch.zeros(embed_dim, r))
    nn.init.normal_(self.A_V, mean=0.0, std=0.02)

  def forward(self, query, key, value):
    """
    query, key, value 的形状为 (batch_size, seq_length, embed_dim)
    """
    # 计算原始的 Q，K，V
    Q = self.W_Q(query)
    K = self.W_K(key)
    V = self.W_V(value)
    
    # 计算 LoRA 增量部分
    delta_Q = torch.matmul(query, self.A_Q.t()) # (batch_size, seq_length,r)
    delta_Q = torch.matmul(delta_Q, self.B_Q.t())  # (batch_size, seq_length, embed_dim)
    delta_K = torch.matmul(key, self.A_K.t())
    delta_K = torch.matmul(delta_K, self.B_K.t())
    delta_V = torch.matmul(value, self.A_V.t())
    delta_V = torch.matmul(delta_V, self.B_V.t())

    # 更新之后的 Q, K, V
    Q = Q + delta_Q
    K = K + delta_K
    V = V + delta_V
    
    # 计算注意力的分
    scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.embed_dim ** 0.5)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    context = torch.matmul(attn_weights, V)
    
    #输出层
    output = self.W_O(context)
    
    return output
```

代码解释：

- 原始权重：W_Q，W_K，W_V 被冻结，不参与训练
- LoRA 参数：A_Q，B_Q，A_K，B_K，A_V，B_V 是可训练的低秩矩阵。
- 前向传播：
  - 首先计算原始的 Q，K，V
  - 然后计算 LoRA 的增量部分，并添加到原始的 Q，K，V 上
  - 接着按照注意力机制进行计算

## 为什么只导入 LoRA 模型不能生图

LoRA 模型只是对原始模型的权重更新进行了低秩近似，存储了权重的增量部分 $\Delta W$, 而不是完整的模型权重 w。

- 仅仅加载 LoRA 模型是无法进行推理的，必须结合原始的预训练模型一起使用。

LoRA 模型就像是给一幅画添加“修改指令”，这些指令需要在原画的基础上才能生效。如果你只有修改指令（LoRA），却没有原画（预训练模型），那么无法得到最终的作品。



