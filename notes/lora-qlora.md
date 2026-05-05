# LoRA / QLoRA 笔记

## 核心思想

LoRA 冻结基础模型权重，只在选定的线性层上训练小型低秩更新矩阵。

对于预训练权重矩阵 `W`，LoRA 学习：

```text
W' = W + BA * alpha / r
```

其中 `A` 和 `B` 是小型可训练矩阵，`r` 是 rank，`alpha` 用来缩放更新量。

## 为什么重要

- 可训练参数量大幅下降。
- 优化器状态占用的显存更少。
- 任务 adapter 更容易保存和切换。
- 对消费级 GPU 更友好，能实践 7B 级模型微调。

## QLoRA 的区别

QLoRA 会以量化形式加载基础模型，常见是 4-bit，同时只训练 LoRA adapter。这样可以显著降低显存需求，让更大的模型能在有限 GPU 上微调。

## 实验想法

- 比较 `r=8`、`r=16`、`r=32`。
- 比较只训练 attention 模块和同时训练 attention + MLP。
- 如果环境支持，比较普通 LoRA 和 4-bit QLoRA。

## 面试问题

- LoRA 为什么省显存？
- LoRA rank 控制什么？
- 量化为什么可能让训练不稳定？
- gradient accumulation 和 effective batch size 是什么关系？
