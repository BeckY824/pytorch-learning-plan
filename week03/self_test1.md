# **🧠 Transformer 自检试卷：注意力机制 · Encoder · Decoder**





> 题型：选择 / 填空 / 简答 / 编程 / 推导题

> 难度：由易到难



------





## **第一部分：注意力机制（易 ➜ 中）**







### **✅ 1. 选择题**





以下哪个不是注意力机制的关键组成部分？

A. Query

B. Key

C. Bias

D. Value



------





### **✅ 2. 填空题**





Scaled Dot-Product Attention 的核心公式为：

$$

\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) \cdot \underline{_______}

$$



------





### **✅ 3. 判断题**





在多头注意力中，各个头的注意力输出是通过加法合并的。（对 / 错）



------





### **✅ 4. 简答题**





简述 “为什么 attention 机制中使用 $\sqrt{d_k}$ 来缩放 $QK^T$？” 有何作用？



------





### **✅ 5. 编程题**





写出一个 PyTorch 的单头注意力实现（不包含 mask），输入形状为 (batch, seq_len, d_model)。



------





## **第二部分：Encoder 模块（中 ➜ 偏难）**







### **✅ 6. 填空题**





Encoder 中，每一层的子结构顺序是：

**Self-Attention → ______ → Add & Norm → FeedForward → ______**



------





### **✅ 7. 选择题**





以下哪一项是 LayerNorm 的归一化维度？

A. batch size

B. sequence length

C. embedding 维度

D. 时间维度



------





### **✅ 8. 简答题**





为什么 Encoder 中的自注意力不使用 mask？它会导致什么问题？



------





### **✅ 9. 推导题**





假设 Encoder 层的输入为 (batch_size, seq_len, d_model)，经过多头注意力机制（num_heads=8）和前馈网络后，输出的维度是多少？

请写出每步维度变化过程。



------





## **第三部分：Decoder 模块（难）**







### **✅ 10. 多选题**





Decoder 中 Masked Multi-Head Attention 的作用包括：

A. 保证自回归特性

B. 加快训练速度

C. 防止模型偷看未来信息

D. 提高注意力的可解释性



------





### **✅ 11. 填空题**





Decoder 层中的第二个 Attention 是 Cross Attention，其中 Query 来自 ______，Key 和 Value 来自 ______。



------





### **✅ 12. 编程题**





你已经实现了 MultiHeadAttention(args, is_causal)，请写出 DecoderLayer.forward(x, enc_out) 中：



- Masked Self Attention
- Cross Attention
- Feed Forward





三个部分的残差连接写法。



------





### **✅ 13. 思维题**





为什么 Decoder 中需要 3 个 LayerNorm，而 Encoder 只需要 2 个？它们分别归一化哪一步？



------





### **✅ 14. 应用题（高阶）**





设计一个只使用注意力机制（不含 CNN/RNN）的句子情感分类器架构。

要求说明每一层的输入、输出以及注意力的作用。



------





## **📌 附加挑战题：**





> ✍️ 推导 Transformer 中位置编码的正余弦公式，并解释它为何对模型有帮助。