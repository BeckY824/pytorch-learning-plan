✅ Day 1：打牢注意力机制 + 位置编码
	•	学习内容：
	•	理解 Q、K、V 的来源和计算
	•	掌握 Scaled Dot-Product Attention 公式
	•	理解 Multi-Head 的作用（并行不同子空间）
	•	位置编码（正余弦 vs 可学习）
	•	代码任务：
	•	实现：
	•	scaled_dot_product_attention(Q, K, V)
	•	MultiHeadAttention
	•	PositionalEncoding
	•	用 toy 数据测试 shape
	•	参照资料：
	•	📘 动手学深度学习 10.1～10.2
	•	🔗 Annotated Transformer 前两节

⸻

✅ Day 2：实现 Encoder Block
	•	学习内容：
	•	理解 Add & Norm 的残差结构
	•	前馈神经网络（FFN）是两层 MLP，ReLU 非线性
	•	编码器堆叠多个相同 Block
	•	代码任务：
	•	实现：
	•	AddNorm（残差连接 + LayerNorm）
	•	PositionwiseFFN
	•	TransformerEncoderBlock
	•	将多个 block 组合成 TransformerEncoder
	•	参照资料：
	•	📘《动手学深度学习》10.3
	•	🔗 PyTorch 官方：TransformerEncoderLayer

⸻

✅ Day 3：实现 Decoder Block（含 Mask）
	•	学习内容：
	•	自回归模型为何需要 Mask（只能看到前面生成的词）
	•	Decoder 中两个注意力：Masked Self-Attention + Cross-Attention
	•	解码器需要保存过去的 key_value，支持推理阶段增量生成
	•	代码任务：
	•	实现：
	•	TransformerDecoderBlock（含掩码）
	•	TransformerDecoder
	•	注意 state 的处理方式：训练 vs 推理
	•	参照资料：
	•	📘《动手学深度学习》10.5
	•	🔗 Annotated Transformer Decoder

⸻

✅ Day 4：组装 Transformer 模型 + Toy 任务准备
	•	学习内容：
	•	理解整体流程：输入嵌入 → 编码器 → 解码器 → 生成预测
	•	学会构造 toy 任务（英文逆序 / copy task）
	•	代码任务：
	•	实现：
	•	Transformer 总模型类
	•	构造小型数据集（如：输入 "i love cat" → 输出 "cat love i"）
	•	编写 Vocab, DataLoader, loss_fn
	•	参照资料：
	•	🔗d2l Transformer 整体代码
	•	🔗Annotated Transformer完整代码

⸻

✅ Day 5：训练 + 推理 + 打印 Attention
	•	学习内容：
	•	梳理训练 loop（teacher forcing）
	•	推理阶段逐词生成（带状态拼接）
	•	可视化或打印 attention weights
	•	代码任务：
	•	编写：
	•	train_epoch(), evaluate(), predict_step()
	•	验证结果正确（loss 下降，输出合理）
	•	打印注意力分数（可以用 heatmap）
	•	参照资料：
	•	🔗 d2l 小任务训练代码
	•	🔗 Attention 可视化文章