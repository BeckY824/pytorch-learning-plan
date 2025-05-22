# 🧠 PyTorch + LLM 实践学习计划

本项目是一个系统性学习计划，旨在掌握训练大语言模型（LLM）、模型对齐、大规模训练、CUDA 编程等核心技术，逐步达成进入前沿 AI 公司的目标。

---

## 🎯 目标技能

- ✅ 熟练使用 PyTorch 进行模型训练
- ✅ 理解 Transformer 架构及其变体
- ✅ 熟悉 LLM 微调与多模态模型（如 CLIP、BLIP）
- ✅ 掌握模型对齐（RLHF、指令微调、偏差控制）
- ✅ 了解大规模分布式训练机制
- ✅ 理解并编写基础 CUDA kernel

---

## 🗓️ 学习计划概览（可持续更新）

| 周数 | 学习主题                                  | 内容关键词 |
|------|-------------------------------------------|-------------|
| Week 01 | PyTorch 基础 + 张量操作                   | `tensor`、`autograd`、`nn.Module` |
| Week 02 | 模型构建 + 训练流程                     | `Dataset`、`DataLoader`、`优化器` |
| Week 03 | Transformer + Attention 理解            | `Self-Attention`、`位置编码` |
| Week 04 | LLM 架构与训练机制                     | `GPT`, `BERT`, `tokenization` |
| Week 05 | 多模态模型：CLIP / BLIP 简介与实践     | `图文对齐`, `对比学习` |
| Week 06 | 模型对齐入门：指令微调 / RLHF 简介      | `alignment`, `preference modeling` |
| Week 07 | CUDA 编程初探 + 大规模训练介绍          | `GPU kernel`, `tensor core`, `DDP` |
| Week 08 | 实战项目：微调一个小 LLM 模型 + 总结    | `finetune`, `evaluation`, `report` |

---

## 📁 项目结构说明

```bash
.
├── week01/
│   ├── tensor_basics.md
│   └── autograd_experiments.ipynb
├── week02/
│   └── training_loop_demo.py
├── transformer/
│   └── attention_from_scratch.py
├── multimodal/
│   └── clip_tutorial.md
├── model-alignment/
│   └── rl_prompt_alignment.md
├── cuda/
│   └── my_first_kernel.cu
├── final_project/
│   └── finetune_tinyllm_report.md
└── README.md
