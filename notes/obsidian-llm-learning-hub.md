# LLM 学习总入口

## 今天从哪里开始

- 当前阶段：第 1-2 周，PyTorch 和深度学习基础
- 当前项目：`projects/pytorch-llm-basics`
- 今天任务：
  - 跑通一个脚本
  - 解释一个训练循环
  - 记录一个不懂的问题

## 学习路线

### 1. PyTorch 基础

- 目标：理解 tensor、autograd、Module、loss、optimizer、训练循环
- 项目：`projects/pytorch-llm-basics`
- 关键问题：
  - `loss.backward()` 做了什么？
  - `.grad` 存在哪里？
  - 为什么要 `optimizer.zero_grad()`？
  - `optimizer.step()` 和学习率有什么关系？

### 2. 文本分类器

- 目标：理解文本如何变成 token id，再变成 embedding，再进入分类器
- 学习记录：[[text-classifier-day2]]
- 关键问题：
  - 为什么文本要先数字化？
  - `Embedding` 是什么？
  - `logits` 是什么？
  - `cross_entropy` 为什么适合分类？

### 3. Tiny Language Model

- 目标：理解 next-token prediction
- 学习记录：[[tiny-lm-day3]]
- 实验记录：[[experiments/tiny-lm-block-dim|Tiny LM block/dim 实验]]
- 关键问题：
  - 输入 token 和目标 token 为什么错开一位？
  - language model 的 loss 在预测什么？
  - batch、sequence length、vocab size 分别在哪里出现？

### 4. Transformer / Mini GPT

- 目标：从零理解 decoder-only Transformer
- 关键问题：
  - self-attention 在算什么？
  - causal mask 为什么必要？
  - residual、LayerNorm、FFN 各自解决什么问题？

### 5. LoRA / QLoRA

- 目标：理解参数高效微调
- 关键问题：
  - LoRA 为什么只训练少量参数？
  - rank 和 alpha 分别影响什么？
  - QLoRA 的量化省了什么显存？

### 6. 推理系统

- 目标：理解吞吐、延迟、KV cache、batching、量化
- 关键问题：
  - prefill 和 decode 有什么区别？
  - KV cache 为什么占显存？
  - batching 为什么可能提高吞吐但增加延迟？

## 每日复盘模板

```text
日期：
今天跑了什么：
看到的关键输出：
我能解释的概念：
我还解释不清的地方：
今天的一个误区：
明天最小下一步：
```

## 概念卡片模板

```text
# 概念：

## 一句话解释


## 生动类比


## 在代码里对应哪里


## 常见误区


## 面试表达


```

## 误区记录

### 误区：`loss.backward()` 会更新参数

- 错在哪里：`backward()` 只计算梯度，不更新参数。
- 正确理解：梯度会存到参数的 `.grad` 中，真正更新参数的是 `optimizer.step()`。
- 记忆句：`backward` 写建议，`step` 动旋钮。

### 误区：梯度每次会自动覆盖

- 错在哪里：PyTorch 默认会累加梯度。
- 正确理解：每一轮训练前通常要调用 `optimizer.zero_grad()` 清空上一轮梯度。
- 记忆句：不擦掉旧红笔，新批改会混在一起。

### 误区：`Linear(4, 1)` 的权重形状是 `(4, 1)`

- 错在哪里：PyTorch 的线性层权重形状是 `(out_features, in_features)`。
- 正确理解：`Linear(4, 1)` 的权重形状是 `(1, 4)`。
- 记忆句：1 个输出神经元，看 4 个输入特征。

## 面试问答库

### Q: PyTorch 的一个基础训练循环包括哪些步骤？

A: 通常包括前向传播、计算 loss、清空梯度、反向传播、优化器更新。前向传播得到预测，loss 衡量预测和目标的差距，`zero_grad()` 清掉上一轮梯度，`backward()` 计算当前梯度，`step()` 根据梯度更新参数。

### Q: `loss.backward()` 和 `optimizer.step()` 的区别是什么？

A: `loss.backward()` 负责沿计算图反向求导，把梯度存到参数的 `.grad` 里；`optimizer.step()` 才会结合学习率和优化器规则，真正修改参数。

### Q: 为什么训练文本模型前要把文本变成数字？

A: 神经网络不能直接处理字符串，它只能处理数字张量。tokenizer 或词表会把文本切成 token，再映射成 token id，后续通过 embedding 把离散 id 变成可学习的向量表示。

## 本周最小交付物

- 能解释 `tensor_autograd.py`
- 能跑通 `text_classifier.py`
- 能解释 `Embedding -> pooling -> Linear -> cross_entropy`
- 能在 `experiments/` 里记录一次学习率或 batch size 改动实验
- 能用自己的话回答 5 个 PyTorch 训练循环问题

## 链接

- [[text-classifier-day2]]
- [[tiny-lm-day3]]
- [[experiments/tiny-lm-block-dim|Tiny LM block/dim 实验]]
- [[lora-qlora]]
- [[transformer-from-scratch]]
- [[llm-inference-systems]]
