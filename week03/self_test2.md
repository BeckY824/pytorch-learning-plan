# **🧠 Transformer & 自注意力机制 自检试卷（理论 + 代码）**

**总共 20 题｜建议用时 90~120 分钟**

涵盖内容：理论理解、PyTorch 实现、模块结构、梯度与高级概念。

------



## **📘 Part 1：理论理解（6 题）**

### **1. 【简答题】**

简述自注意力机制中 Query（Q）、Key（K）、Value（V）三者的含义与作用。

Query是用来查询的量，Key和Value是对应的键与值。

三者是作用在自注意力机制中的，首先Query和Key用来计算一个权重，最后得到的值乘以V得到自注意力值。

**补充**：

Q = 查询向量， K = 键向量，V = 值向量。Attention机制根据Q与所有K的相似度计算加权系数，再对所有V加权求和，得到输出。

------



### **2. 【推导题】**

假设有一个输入序列张量 X = [x1, x2, x3]，请说明如何从 X 生成自注意力机制所需的 Q、K、V，并写出 Attention 的计算公式全过程（包括 softmax 权重的计算）。

```
提示：你可以用 dot-product 的注意力表示方式写出：
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

```
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
scores = Q @ K.T / sqrt(d_k)
attn_weights = softmax(scores)
out = attn_weights @ V
```

------





### **3. 【判断题】**

判断下列说法是否正确，并简要说明理由。

```
a) 多头注意力的主要目的是降低计算复杂度。错误

b) 自注意力机制相比 RNN 更高效的一个原因是它可以并行处理序列中的所有位置。正确
```



------



### **4. 【填空题】**

Transformer 中的残差连接的作用是 _提高拟合避免梯度消失，保留更多原始特征__，通常与 _attention & MLP____ 层一起使用，以稳定训练。

**补充：**

作用：保留原始特征，便于梯度流动，避免退化 

配合：LayerNorm使用

------



### **5. 【简答题】**

为什么需要 Position Embedding？请列出两种不同的实现方式并简要说明它们的差别。

1.因为Transformer的自注意力机制是无关词序的，所以在自注意力看来 i like you 和 you like i 是一样的。但实际上是不一样的。所以需要位置编码嵌入。

2.我只会一种是关于使用sin & cos的位置表示方法

**补充：**

另一种方法为
Learnable Position Embedding，训练得到的可学习向量，表现比正余弦更灵活。

------



### **6. 【应用题】**

请简要说明 Transformer 能够捕捉长距离依赖的原因，它是如何克服传统 RNN 中梯度消失问题的？

1.多头并行处理（多头自注意力机制）

2.使用dropout

**补充：**

自注意力可以直接建立任意位置之间的联系，权重计算中没有时间步的限制，因此能捕捉长依赖。

------



## **💻 Part 2：代码理解与实现（6 题）**



### **7. 【填空题】**

以下是一个简化的单头自注意力机制，请补全 scores, attn_weights, out 的计算：

```
def single_head_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = _torch.matmul(Q,K.T)/math.sqrt(d_k)___       # Q @ K^T / sqrt(d_k)
    attn_weights = _F.softmax(scores)___ # softmax over scores
    out = _torch.matmul(attn_weights,V)          # attn_weights @ V
    return out
```

**补充：**

1.K.T应该是 K.transpose(-2,-1),

2.F.softmax(scores, dim=-1)

------



### **8. 【理解题】**

阅读下方 PyTorch 多头注意力代码：

```
attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
output, attn_weights = attn(query, key, value)
```

回答以下问题：

- a) 如何指定注意力头的数量？ num_heads = n
- b) 输入 shape 是 (seq_len, batch_size, embed_dim)，输出 shape 是？

 (batch_size,seq_len, embed_dim)

**补充：**

 (seq_len, batch_size, embed_dim) 与输入保持一致

------



### **9. 【实现题】**

请用 PyTorch 手动实现一个完整的 MultiHeadAttention 模块，要求包括以下结构：

```
- 输入线性映射生成 Q, K, V
- 按头拆分（reshape）
- 计算注意力权重
- 拼接所有头的输出
```

你可以写成类结构或函数均可，要求尽量简洁明了。

还是有点不清晰这一块。

```python
def MultiHeadAttention(nn.Module):
	def __init__(self, embed_dim, num_heads):
    super().__init__()
    assert embed_dim % num_heads == 0
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    
    self.q_proj = nn.Linear(embed_dim, embed_dim)
    self.k_proj = nn.Linear(embed_dim, embed_dim)
    self.v_proj = nn.Linear(embed_dim, embed_dim)
    self.out_proj = nn.Linear(embed_dim, embed_dim)
    
  def forward(self,x):
    # x:(batch_size, seq_len, embed_dim)
    B,T,C = x.size()
    
    # step1:Linear projections
    Q = self.q_proj(x)
    K = self.k_proj(x)
    V = self.v_proj(x)
    
    # step2:Reshape for multi-head
    def reshape(x):
      return x.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
    Q,K,V = reshape(Q), reshape(K),reshape(V)
    
    # step3:scaled dot-product attention
    scores = Q @ K.transpose(-2,-1) / (self.head_dim ** 0.5)
    attn_weights = F.softmax(scores,dim=-1)
    attn_output = attn_weights @ V
    
    # step4:concatente heads
    attn_output = attn_output.transpose(1,2).contiguous().view(B,T,C)
    
    return self.out_proj(attn_output)
    
 
```

🧠 **核心知识点**：

- 多头的实质是“多组不同 Q/K/V 的线性映射 + 拼接”
- 使用 .view() + .transpose() 拆分 heads
- 拼接后 .contiguous().view() 恢复 shape

------





### **10. 【填空题】**

Transformer 中 LayerNorm 的位置通常是在 ___后_________（前/后），称为 Post-LN；也有变体使用 Pre-LN。请说明这两种方式的主要差异及各自优劣。

LayerNorm的存在就是为了统一数据的分布。具体的差异我记得不是很清晰。

**补充：**

Post-LN，残差后再 LayerNorm，训练初期稳定但难以深层训练

Pre- LN，残差前 LayerNorm，训练更容易收敛，适合深层模型

------



### **11. 【调试题】**

以下 TransformerBlock 中存在一个潜在 Bug，请指出并说明可能带来的问题：

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim):
        self.attn = MultiHeadSelfAttention(dim)
        self.ffn = FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

**补充：**

- super().__init__()

- ```python
  class TransformerBlock(nn.Module):
      def __init__(self, dim):
          self.attn = MultiHeadSelfAttention(dim)
          self.ffn = FeedForward(dim)
          self.norm1 = nn.LayerNorm(dim)
          self.norm2 = nn.LayerNorm(dim)
  
      def forward(self, x):
          x = x + self.attn(self.norm1(x))
          x = x + self.ffn(self.norm2(x))
          return x
  ```

  

------



### **12. 【分析题】**

训练 Transformer 时发现 loss 长时间不下降，学习率正常，模型结构未变。请列出 3 个可能原因，并给出初步调试建议。

1.初始层数过深（减少层数）

其余方法暂时想不出来。

**补充：**

2.初始化不当（如bias全为0）

3.学习率调度无效

4.dropout过大

------



## **🧩 Part 3：结构细节记忆（4 题）**

### **13. 【简答题】**

Transformer 的 Encoder 和 Decoder 分别由哪些模块构成？请指出它们的结构差异，特别是在注意力机制的使用上。

Encoder是由 attention, layer norm ,fnn, layer norm  dropout组成

Decoder是由 mask attention, self attention fnn , dropout组成

Decoder多一个mask attention，用来处理query的输入。

------



### **14. 【结构题】**

请按顺序写出 Transformer Encoder Layer 内部模块的计算流程，并描述每一步的张量 shape 变化（假设输入为 (batch, seq_len, dim)）。

batch, seq_len, dim ->

batch, seq_len, dim/n_head ->

batch, seq_len, dim

**补充：**

(batch, seq_len, dim) -> Q/K/V: (batch, n_heads, seq_len, dim/n)
-> scores -> attention -> (batch, n_heads, seq_len, dim/n)
-> concat -> (batch, seq_len, dim)

```
Input: x (B, T, D)

1. LayerNorm
2. Multi-Head Attention
   - Q/K/V Linear Proj: (B, T, D) -> (B, T, D)
   - Reshape: (B, T, D) -> (B, n_heads, T, head_dim)
   - Attention: (B, n_heads, T, T)
   - Output: (B, n_heads, T, head_dim) -> (B, T, D)
3. Dropout + Residual
4. LayerNorm
5. Position-wise FFN
   - Linear + ReLU + Linear
   - (B, T, D) -> (B, T, D)
6. Dropout + Residual
```

------



### **15. 【选择题】**

以下哪些属于 Transformer 的关键组件？（可多选）B D E

```
A. GRU 单元  
B. Multi-Head Attention  
C. Position-wise FFN  
D. Dropout  
E. LayerNorm
```

**补充：**

C也是。

------



### **16. 【整理题】**

请从输入的 token 序列出发，整理 Transformer 处理每个 token 时经历的主要步骤，包括编码结构、矩阵运算和最终输出生成。

token -> token embedding -> token embedding + positional embedding -> encoder ->  attention计算 -> layer norm -> fnn -> layer norm -> dropout - > enc_out -> decoder -> mask attention, attention, fnn -> dropout -> output -> MLP -> out



**补充：**

token -> embedding -> position embedding -> encoder stack
-> 输出 context 表示 -> decoder stack (含 masked attn + cross attn)
-> 输出 -> linear proj -> softmax -> logits

```
Input: token_ids -> embedding (lookup) -> shape: (B, T, D)

1. Add positional encoding (learned or sinusoidal)
2. 输入送入 N 层 EncoderBlock：
   - 每层包含：Multi-Head Attention + FFN + 残差 + LayerNorm
3. 得到 Encoder 输出（上下文表征）

Decoder 流程（语言建模或翻译）：

4. Decoder 端输入（target 端）
   - Masked Self-Attention（阻止看到未来）
   - Cross-Attention（使用 Encoder 输出作为 K,V）
   - FFN + 残差 + LayerNorm
5. 最终输出 Logits（通过线性 + softmax）
```

------



## **🔬 Part 4：推导与进阶（4 题）**

### **17. 【推导题】**

请推导自注意力机制中输出对 V 的梯度公式（你可以仅推导 softmax(QK^T / sqrt(d_k)) * V 中 V 的导数部分）。

softmax(QK^T / sqrt(d_k))

**补充：**

dL/dV = softmax(QK^T / sqrt(d_k))^T @ dL/dout

------



### **18. 【分析题】**

在多头注意力机制中，多个 head 之间共享什么？不共享什么？为什么拼接各个 head 的结果而不是直接相加？

多个head之间共享已经创造的token，而不共享未知的token（predict）。

相加就无法比较最佳结果。

**补充：**

1.共享输入，不共享 Q/K/V 的线性映射；

2.拼接而非相加，是为了保留多种注意力模式。

------



### **19. 【比较题】**

请比较标准自注意力（vanilla self-attention）与 masked self-attention 的不同点，并说明后者在 Transformer Decoder 中的作用。

不同的点在于：
1.训练时期的不同，mask会随机遮挡15%左右的信息，让attention去预测。
2.不共享未知的token信息。

**补充：**

Masked self-attn 是为避免 decoder 看到未来 token；
训练阶段避免信息泄露，是 causal attention 的一种。

| **类型** | **Vanilla Self-Attn**  | **Masked Self-Attn**              |
| -------- | ---------------------- | --------------------------------- |
| 用途     | Encoder 中全局依赖建模 | Decoder 中防止未来信息泄漏        |
| 掩码     | 无限制，可访问所有位置 | 仅可访问当前及前面 token          |
| 公式区别 | 无 Mask                | 加上 mask 后的 score 被 -inf 处理 |

------



### **20. 【开放题】**

有研究认为 attention matrix 中很多值接近 0，提出了稀疏注意力（如 Top-k attention）机制。你怎么看待这种优化方式？是否会带来性能提升？可能存在哪些风险或挑战？

这种做法很像 pca。

补充：

减少计算量，提升稀疏性；风险：破坏信息流、可导性差。

------

## **✅ 小结与建议**

| **模块** | **得分**  | **建议**                               |
| -------- | --------- | -------------------------------------- |
| 理论理解 | 11.5 / 30 | 多看一遍 Attention 全公式及其原理      |
| 代码理解 | 9.5 / 30  | 建议从头撸一遍 MultiHeadAttention 类   |
| 模块结构 | 9 / 20    | 多画图，记住 Encoder/Decoder 流程图    |
| 高阶推导 | 4 / 20    | 需要加强推导、可视化梯度与优化技巧理解 |