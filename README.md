# LLM 实习冲刺学习工作区

这个工作区把 12 周的 LLM 训练、微调、推理系统学习计划变成可以每天执行的材料。

目标不是“刷完很多课程”，而是做出能支撑 LLM 实习面试的作品集：

- 能讲清楚 PyTorch 和 Transformer 基础。
- 从零实现一个 mini GPT。
- 完成一个 LoRA/QLoRA 微调项目，并做真实评测。
- 基于双 RTX 4090 服务器完成一个推理性能 benchmark。
- 写出技术笔记和简历 bullet，让你的工作能被招聘方快速看懂。

## 如何使用这个工作区

每个工作日早上：

1. 从 `templates/daily-log.md` 复制一份到 `daily/YYYY-MM-DD.md`。
2. 只学习当天任务需要的概念。
3. 用 100 分钟写代码、调试或分析实验。
4. 上班前启动一次训练、评测或 benchmark。
5. 记录命令、配置、预期结果和下一步。

每周五：

1. 填写 `templates/weekly-review.md`。
2. 更新当前项目的 README。
3. 把有价值的实验结果整理到 `experiments/`。
4. 如果本周有可量化成果，就补充一条简历 bullet。

## 目录结构

```text
.
├── curriculum/                 # 12 周学习计划和资料地图
├── daily/                      # 每日学习记录
├── experiments/                # 可复现实验记录和结果表
├── notes/                      # LoRA、QLoRA、推理系统、论文笔记
├── projects/
│   ├── pytorch-llm-basics/      # 第 1-2 周
│   ├── mini-gpt-from-scratch/   # 第 3-4 周
│   ├── llm-lora-finetuning/     # 第 5-7 周
│   └── llm-inference-benchmark/ # 第 8-10 周
├── career/                     # 简历 bullet 和面试准备
└── templates/                  # 日志、周总结、实验、论文模板
```

## 每周里程碑

| 周次 | 重点 | 交付物 |
|---|---|---|
| 第 1 周 | PyTorch 基础 | 张量、autograd、Module 练习 |
| 第 2 周 | 训练循环 | 文本分类器和 tiny LM 训练脚本 |
| 第 3 周 | Transformer 内部机制 | attention、mask、block 实现 |
| 第 4 周 | Mini GPT | 训练并采样一个小型 GPT |
| 第 5 周 | Hugging Face 生态 | 数据集、tokenizer、模型加载、baseline |
| 第 6 周 | LoRA/QLoRA | PEFT 微调实验和训练曲线 |
| 第 7 周 | 评测 | 微调报告和错误分析 |
| 第 8 周 | 推理基础 | 延迟/吞吐测量脚本 |
| 第 9 周 | vLLM/量化 | 推理后端和量化对比 |
| 第 10 周 | 双 4090 benchmark | 可复现的性能报告 |
| 第 11 周 | 技术写作 | 两篇作品集技术笔记 |
| 第 12 周 | 面试包装 | 简历 bullet 和面试回答 |

## 默认技术栈

- Python 3.10+
- PyTorch
- Transformers
- Datasets
- Accelerate
- PEFT
- bitsandbytes
- vLLM 或 llama.cpp
- TensorBoard 或 Weights & Biases

只使用公开数据集或脱敏样例。不要提交公司数据、密钥、包含私有 prompt 的日志、不可分发的模型权重，或未经允许的内部 benchmark 结果。
