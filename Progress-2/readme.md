# **🗂️ 4个月学习+实践路线图**

## **第1个月：大模型基础 + 编程夯实**

🔑 目标：理解 LLM 基本原理，能跑通并改造一个小规模大模型。

- **理论**
  - Transformer 全流程复习（Attention → Encoder/Decoder → Masking → FFN → LayerNorm）
  - 阅读：《动手学深度学习》 + Annotated Transformer（代码实现版）
  - 微调方法原理：LoRA、Prefix-tuning、QLoRA
- **实战**
  - 跑通 HuggingFace 的小模型（如 distilGPT2, Qwen1.5-1.8B）
  - 尝试 LoRA 微调中文任务（唐诗/分类/对话微调）
  - 学会使用 transformers + peft + deepspeed

📌 **本月成果**：

1. 个人笔记：《Transformer 100 问》
2. 复现 + 微调一个小规模 GPT 模型（能展示在 GitHub）

## **第2个月：多模态入门 + RAG 系统**

🔑 目标：掌握多模态对齐思路，能搭建文本+图像/文本+知识库的检索增强系统。

- **理论**
  - 多模态模型代表作：CLIP、BLIP-2、LLaVA、Qwen-VL
  - 多模态 slow thinking 概念（推理链、分步理解）
  - RAG 技术栈：向量数据库（FAISS, Milvus, Weaviate） + LLM
- **实战**
  - 复现 CLIP，做“图文相似度搜索”demo
  - 构建一个 RAG Demo（如：上传PDF → LLM问答）
  - 尝试向量数据库（FAISS → Milvus）

📌 **本月成果**：

1. Demo1：图片搜索助手（图文检索）
2. Demo2：文档问答（RAG系统）

## **第3个月：多模态生成 + 工程化**

🔑 目标：进入文生图/语音等生成模型，掌握推理加速与部署。

- **理论**
  - Diffusion 模型（Stable Diffusion, ControlNet 原理）
  - 语音处理：ASR（Whisper）、TTS（VITS, CosyVoice）
  - 模型部署优化：ONNX、TensorRT、vLLM
- **实战**
  - 复现 Stable Diffusion 文生图（跑本地模型，尝试 LoRA finetune）
  - 加入 ControlNet 实现“文生图 + 条件约束”
  - 部署 Whisper 做语音识别 + LLM对话
  - 学习使用 vLLM 提升推理性能

📌 **本月成果**：

1. Demo3：文生图助手（Stable Diffusion + ControlNet）
2. Demo4：语音对话助手（Whisper + LLM）

## **第4个月：综合应用 + 面试准备**

🔑 目标：把大模型、多模态、RAG整合成一个综合项目，并准备面试材料。

- **理论**
  - Agent 框架：LangChain、LlamaIndex
  - 大模型 slow thinking 应用案例
  - 模型评估与指标：BLEU、ROUGE、CLIP-score、Human Eval
- **实战（综合项目）**
  - 构建 **多模态智能助手**：
    - 输入：文字 + 图片/语音
    - 能做：检索（RAG）+ 生成（回答/文生图/语音合成）
  - 尝试接入 LangChain 或 LlamaIndex
- **面试准备**
  - 整理 GitHub 项目，写好 README
  - 总结 30 道高频面试题（Transformer、LoRA、RAG、多模态）
  - 模拟系统设计题：如果让你做一个“AI助理App”，如何落地？

📌 **本月成果**：

1. Final Project：多模态智能助手（展示综合能力）
2. 完整的面试准备文档 + GitHub项目集

# **🚀 输出成果（简历可写）**

- GitHub 上至少有 **3-4 个小项目 + 1 个综合项目**
- 技术栈覆盖：**Transformer / LoRA / RAG / CLIP / Stable Diffusion / Whisper / LangChain**
- 形成一套 **学习笔记 + 面试问答集**

