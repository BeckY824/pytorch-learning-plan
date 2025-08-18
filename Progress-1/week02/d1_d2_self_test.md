✅ Transformer 自测问卷（10.1–10.6 + 编码器）

🧠 一、基础理解题（填空 / 选择）
	1.	填空：注意力机制的核心思想是__________地聚合输入信息，以强调与当前任务相关的部分。
	2.	填空：缩放点积注意力（scaled dot-product attention）公式为
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
其中 \sqrt{d_k} 的作用是：____________________。
	3.	单选题：以下哪个是“缩放点积注意力”的优势？
	•	A. 参数少，训练更快
	•	B. 计算更稳定，梯度更平滑
	•	C. 不需要学习权重矩阵
	•	D. 不需要 mask 操作
	4.	判断题（✔/✘）：多头注意力的主要作用是让模型从多个子空间并行学习不同的注意力模式。_____
    	5.	填空：位置编码的作用是：___________________________________________。

⸻

✍️ 二、公式与推导题
	1.	请推导缩放点积注意力中 softmax 前的得分矩阵 shape，以及最终输出的 shape，假设：
	•	Query shape: (batch_size, q_len, d)
	•	Key shape: (batch_size, k_len, d)
	•	Value shape: (batch_size, k_len, d)
	2.	解释“掩码机制（masking）”在训练中的必要性，举例说明在机器翻译任务中如何使用掩码。

⸻

🧪 三、代码理解题
	1.	阅读以下代码，指出它实现的是哪一部分功能？输出形状是什么？
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d)
    attention_weights = nn.functional.softmax(scores, dim=-1)
    output = torch.bmm(attention_weights, V)

    2.	判断下列模块在 Transformer 编码器中的顺序是否正确：
    输入 → 多头注意力 → 残差连接 → LayerNorm → 前馈网络 → 残差连接 → LayerNorm

    3.	填空：前馈网络 FFN 中常用的激活函数是 _________，它的结构是：

    \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2

🎯 四、思维与应用题
	1.	简述自注意力（Self-Attention）与传统 CNN 中的卷积在建模远距离依赖上的差异。
	2.	如果你设计一个 Transformer 模型处理时间序列（例如天气预测），你会如何设计位置编码？是否适合使用可学习位置编码？为什么？
	3.	对于文本分类任务，你准备只使用 Transformer 的编码器部分而不使用解码器，说明你的理由。
