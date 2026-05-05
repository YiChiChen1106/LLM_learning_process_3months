# 12 周 LLM 训练、微调、推理系统冲刺计划

## 执行条件

- 时间：每周 5 天，每天早上 3 小时专注学习。
- 算力：双 RTX 4090 服务器，默认可以长期使用。
- 目标：形成 LLM 实习作品集。
- 主方向：训练、微调、推理系统。

## 每日节奏

| 时间 | 任务 |
|---|---|
| 00:00-00:40 | 阅读一个概念、课程片段、论文小节或文档页面 |
| 00:40-02:20 | 写代码、调试或分析当天任务 |
| 02:20-02:50 | 启动训练、评测或 benchmark |
| 02:50-03:00 | 记录命令、假设、预期结果和下一步 |

## 第 1-2 周：PyTorch 和深度学习基础

目标：

- 理解 tensor、autograd、Module、optimizer、loss 和训练循环。
- 建立阅读和修改小型 PyTorch 程序的信心。
- 理解基本显存概念：参数、激活值、梯度、优化器状态。

交付物：

- `projects/pytorch-llm-basics`
- 文本分类器。
- tiny next-token language model 训练脚本。
- loss 曲线和简短解释。

验收标准：

- 能解释 forward、backward、batch size、learning rate、overfitting、validation。
- 能不照抄模板地修改一个训练循环。

## 第 3-4 周：Transformer 和 Mini GPT

目标：

- 理解 tokenization、embedding、self-attention、MHA、FFN、残差连接、LayerNorm 和 causal mask。
- 实现一个最小 decoder-only Transformer。
- 在小语料上训练并生成文本。

交付物：

- `projects/mini-gpt-from-scratch`
- 训练脚本。
- 采样脚本。
- 解释模型结构和结果的 README。

验收标准：

- 能画出并解释一个 Transformer block。
- 能从概念上解释 prefill、decode 和 KV cache。

## 第 5-7 周：LoRA / QLoRA 微调

目标：

- 学会 Transformers、Datasets、Tokenizers、Trainer/SFTTrainer、Accelerate、PEFT 和 bitsandbytes。
- 在可行时对 7B 级开源模型做 baseline 推理和 LoRA/QLoRA 训练。
- 产出评测结果，而不是只保存一个 adapter。

交付物：

- `projects/llm-lora-finetuning`
- baseline 结果。
- LoRA/QLoRA 配置。
- 训练曲线。
- 评测表格。
- 错误案例分析。

验收标准：

- 能解释 LoRA rank、alpha、target modules、量化、effective batch size 和 gradient accumulation。
- 能用一条清晰命令复现一次训练。

## 第 8-10 周：推理系统与 Benchmark

目标：

- 理解 prefill/decode、KV cache、batching、量化、吞吐、延迟和显存压力。
- 至少比较两种推理方式。
- 用双 4090 服务器产出可信的 benchmark 报告。

交付物：

- `projects/llm-inference-benchmark`
- benchmark 脚本。
- 结果表。
- 图表。
- 文字结论。

验收标准：

- 能解释为什么 batching 可能提高吞吐但增加延迟。
- 能估算为什么长上下文会增加 KV cache 显存。
- 能说明 vLLM、Transformers、llama.cpp 分别适合什么场景。

## 第 11-12 周：作品集与面试包装

目标：

- 把实验变成别人看得懂的作品。
- 准备面试讲解。
- 写出有数据支撑的简历 bullet。

交付物：

- `notes/` 里的两篇技术笔记。
- 两个主项目的最终 README。
- `career/resume-bullets.md` 中的简历 bullet。
- `career/interview-prep.md` 中的面试问答库。

验收标准：

- 招聘方能从 README 快速看懂项目价值。
- 工程师能按说明复现关键实验。
- 你能解释简历中每一个数字。
