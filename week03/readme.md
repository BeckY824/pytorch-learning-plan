# 🚀 5 天 Transformer 手写实现计划（基于 Happy LLM 教程）

> 🎯 目标：5 天内通过 Happy LLM Chapter 2 完成 Transformer 的从理解到手写实现  
> 📚 教程参考：[Happy LLM - Transformer 架构](https://datawhalechina.github.io/happy-llm/#/./chapter2/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Transformer%E6%9E%B6%E6%9E%84)  
> 🛠️ 输出成果：完整代码实现（encoder, decoder, attention, transformer）+ 可运行 toy demo

---

## 🗓️ Day 1 - 理解 Transformer 结构总览 + Embedding 与 Positional Encoding

### ✅ 学习目标：
- 理解 Transformer 架构中 encoder / decoder 的总体结构
- 理解 Token Embedding、位置编码（PositionalEncoding）机制
- 掌握输入嵌入的尺寸变换

### 📘 学习内容：
- Happy LLM 教程 2.1 ～ 2.3 节  
  - `2.1 Transformer结构解析`
  - `2.2 Embedding层实现`
  - `2.3 位置编码层实现`

### ✏️ 编码任务：
- 实现 `TokenEmbedding` 类
- 实现 `PositionalEncoding` 类（使用 sin/cos）
- 编写简单的测试样例，验证输出 shape 正确

---

## 🗓️ Day 2 - 理解注意力机制（Self-Attention）与 Multi-Head Attention

### ✅ 学习目标：
- 推导并理解 Scaled Dot-Product Attention
- 实现多头注意力机制（MultiHeadAttention）

### 📘 学习内容：
- Happy LLM 教程 2.4 ～ 2.5 节  
  - `2.4 Scaled Dot-Product Attention`
  - `2.5 多头注意力机制`

### ✏️ 编码任务：
- 实现 `ScaledDotProductAttention` 类
- 实现 `MultiHeadAttention` 类
- 使用随机输入，验证注意力 shape 和行为

---

## 🗓️ Day 3 - 实现前馈网络、残差连接、LayerNorm 与 Encoder Block

### ✅ 学习目标：
- 掌握 Add & Norm（残差 + LayerNorm）
- 实现 Position-wise FFN
- 构建一个完整的 Encoder Block 模块

### 📘 学习内容：
- Happy LLM 教程 2.6 ～ 2.8 节  
  - `2.6 前馈神经网络`
  - `2.7 Add & Norm`
  - `2.8 TransformerEncoderBlock 实现`

### ✏️ 编码任务：
- 实现 `PositionwiseFeedForward`
- 实现 `AddNorm`
- 实现 `TransformerEncoderBlock` 类

---

## 🗓️ Day 4 - 理解 Masked Attention 与 Decoder 构建

### ✅ 学习目标：
- 理解 Decoder 中 Masked Self-Attention 的作用
- 理解 Encoder-Decoder Attention 的机制
- 实现 Transformer Decoder Block

### 📘 学习内容：
- Happy LLM 教程 2.9 ～ 2.10 节  
  - `2.9 Mask机制`
  - `2.10 TransformerDecoderBlock 实现`

### ✏️ 编码任务：
- 实现 Mask 矩阵生成函数
- 实现 `TransformerDecoderBlock` 类
- 处理好 state 拼接（训练 vs 推理）

---

## 🗓️ Day 5 - 拼装完整模型 + Toy 任务验证

### ✅ 学习目标：
- 拼装完整 Transformer 模型（含 encoder, decoder, embedding）
- 构造 toy 数据集进行验证（如逆序任务）

### 📘 学习内容：
- Happy LLM 教程 2.11 ～ 2.12 节  
  - `2.11 Transformer模型整合`
  - `2.12 完整代码演示`

### ✏️ 编码任务：
- 实现 `Transformer` 模型类
- 编写前向推理函数 `forward()`
- 构造 toy 数据（输入 + 输出对）
- 跑通 forward + loss 计算 + 预测

---

## 📦 附加建议

- 每天阅读结束后，写一小段总结笔记
- 每段代码写完后都加 shape 注释（极有帮助）
- 可以将最终代码 push 到 GitHub，作为项目记录

---

## ✅ 最终成果

- ✅ 完整 Transformer 实现：Embedding, Attention, FFN, Encoder, Decoder, Model
- ✅ 能运行 toy 示例，理解输入输出流程
- ✅ 对每个模块的作用有清晰理解

---

📍 学完本计划后，你将具备从零构建 Transformer 的能力，为 LLM 预研打下扎实基础。