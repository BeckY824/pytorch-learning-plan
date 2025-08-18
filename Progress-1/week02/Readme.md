# 📘 Transformer 学习计划 - 第二周

> 🎯 本周目标：掌握 Transformer 编码器结构，深入理解自注意力机制、前馈网络与残差连接，并手动实现简化的 EncoderBlock。

---

## 🧠 学习主题

### 1. Transformer 基础结构复习
- [x] Encoder / Decoder 架构整体回顾
- [x] EncoderBlock 模块组成（Self-Attention → Add & Norm → FFN → Add & Norm）

### 2. 自注意力机制（Self-Attention）
- [x] 三输入含义：Query、Key、Value
- [x] 多头注意力机制原理（Multi-Head Attention）
- [x] 点积注意力的计算流程与掩码机制（masking）

### 3. 前馈神经网络（Position-wise FFN）
- [x] 为什么每个位置都要单独经过 FFN
- [x] 非线性变换对特征表达能力的提升作用

### 4. 残差连接与 LayerNorm
- [x] 为什么要使用残差连接？如何提升稳定性？
- [x] LayerNorm 和 BatchNorm 的对比理解

---

## 💻 实践任务

> 推荐使用 PyTorch，仿照李沐的《动手学深度学习》风格编写模块。

### ✅ 编码任务
- [x] 实现 AddNorm 模块（Add + LayerNorm + Dropout）
- [x] 实现 PositionWiseFFN（含两个线性层 + ReLU）
- [x] 实现 EncoderBlock（含 MultiHeadAttention + AddNorm + FFN）

### 🧪 测试任务
- [ ] 编写单元测试验证 EncoderBlock 的输入输出维度正确
- [ ] 测试模型在 batch 输入下的 forward 正常运行

---

## 📚 推荐资料

### 📘 核心阅读材料
- [ ] 《动手学深度学习》第11章：注意力机制（重点：11.6 Transformer）
- [ ] [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)（英文原理清晰）
- [ ] PyTorch 官方 MultiheadAttention 文档

### 🎥 推荐视频
- [ ] 李沐 Transformer 教学视频（B站）
- [ ] Stanford CS224n 2023 - Attention & Transformers 课时

---

## 🏁 本周目标检验（周末回顾）

- [ ] 能够手写 EncoderBlock 并清晰注释每个子模块功能
- [ ] 能准确解释 self-attention 中为什么 Q=K=V
- [ ] 理解前馈网络对每个位置的作用
- [ ] 明确残差连接如何帮助深层网络收敛

---

## 📎 附加挑战（可选）

- [ ] 尝试实现位置编码（Positional Encoding）
- [ ] 扩展 EncoderBlock 为 N 层堆叠的 Encoder

---

📅 时间建议：每天 1～1.5 小时