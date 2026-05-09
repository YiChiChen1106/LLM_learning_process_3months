# LLM 学习总入口

## 今天从哪里开始

- 当前阶段：第 1-2 周，MiniGPT from scratch
- 当前项目：`projects/mini-gpt-from-scratch`
- 今天任务：
  - 复习 `train.py` 的 validation loss 评估流程
  - 解释 `best checkpoint / final checkpoint / early stopping`
  - 用自己的话总结为什么训练不能只看 train loss

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
- 学习记录：[[2026-05-06]]、[[2026-05-07]]、[[2026-05-08]]、[[2026-05-09]]
- 关键问题：
  - self-attention 在算什么？
  - causal mask 为什么必要？
  - residual、LayerNorm、FFN 各自解决什么问题？
  - train loss 和 validation loss 如何帮助判断过拟合？
  - 采样时为什么只取 `logits[:, -1, :]`？
  - temperature、top-k、top-p 分别改变了什么？
  - `torch.multinomial`、`seed` 和 greedy decoding 有什么区别？
  - `model.eval()` 和 `torch.no_grad()` 有什么区别？
  - best checkpoint 和 final checkpoint 有什么区别？
  - early stopping 的 patience 是什么？

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

### 误区：低 train loss 就代表模型泛化好

- 错在哪里：train loss 只说明模型在参与训练的数据上表现好，不能说明它对没见过的数据也好。
- 正确理解：需要看 validation loss。如果 train loss 持续下降但 val loss 上升，通常是过拟合信号。
- 记忆句：训练集考满分，不代表换一张卷子也会。

### 误区：sampling 策略会让模型变聪明

- 错在哪里：temperature、top-k、top-p、greedy 都不改变模型参数。
- 正确理解：采样策略只改变从 logits 或概率分布里选择下一个 token 的方式，会影响稳定性、多样性和随机性。
- 记忆句：采样调的是选法，不是脑子。

### 误区：`model.eval()` 等于不记录梯度

- 错在哪里：`model.eval()` 只切换 dropout、batchnorm 等模块行为，不负责关闭梯度追踪。
- 正确理解：推理时通常同时使用 `model.eval()` 和 `with torch.no_grad():`。
- 记忆句：`eval` 管行为，`no_grad` 管计算图。

### 误区：`with torch.no_grad` 可以少写括号

- 错在哪里：`torch.no_grad` 没有被调用，不能作为上下文管理器使用。
- 正确理解：正确写法是 `with torch.no_grad():`。
- 记忆句：no_grad 要开门，括号就是门把手。

### 误区：最后一步模型就是最好的模型

- 错在哪里：最后一步模型可能 train loss 更低，但 validation loss 更高。
- 正确理解：best checkpoint 应该保存 validation loss 历史最低的模型。
- 记忆句：最后交卷的人，不一定分最高。

### 误区：early stopping 停止点就是 best checkpoint

- 错在哪里：early stopping 只是停止继续训练，停止那一步不一定是验证集表现最好的模型。
- 正确理解：best checkpoint 是历史最低 validation loss 对应的模型，early stopping 是连续多次没有改善后的停止动作。
- 记忆句：刹车点不是最高分点。

## 面试问答库

### Q: PyTorch 的一个基础训练循环包括哪些步骤？

A: 通常包括前向传播、计算 loss、清空梯度、反向传播、优化器更新。前向传播得到预测，loss 衡量预测和目标的差距，`zero_grad()` 清掉上一轮梯度，`backward()` 计算当前梯度，`step()` 根据梯度更新参数。

### Q: `loss.backward()` 和 `optimizer.step()` 的区别是什么？

A: `loss.backward()` 负责沿计算图反向求导，把梯度存到参数的 `.grad` 里；`optimizer.step()` 才会结合学习率和优化器规则，真正修改参数。

### Q: 为什么训练文本模型前要把文本变成数字？

A: 神经网络不能直接处理字符串，它只能处理数字张量。tokenizer 或词表会把文本切成 token，再映射成 token id，后续通过 embedding 把离散 id 变成可学习的向量表示。

### Q: train loss 和 validation loss 的区别是什么？

A: train loss 是模型在参与参数更新的数据上的损失，validation loss 是模型在不参与训练的数据上的损失。低 train loss 只能说明模型拟合了训练集，validation loss 才能帮助判断模型是否具备一定泛化能力。

### Q: temperature、top-k 和 top-p 分别控制什么？

A: temperature 在 softmax 前缩放 logits，控制概率分布的尖锐或平滑程度；top-k 固定保留概率最高的 k 个 token；top-p 按概率从高到低累加，动态保留累计概率达到 p 的候选集合。

### Q: `seed` 和 greedy decoding 的区别是什么？

A: seed 固定随机数生成过程，让 `multinomial` 这种随机采样可复现，但它仍然是在概率分布里抽样；greedy decoding 不随机抽样，每一步直接选择概率最大的 token。

### Q: `model.eval()` 和 `torch.no_grad()` 的区别是什么？

A: `model.eval()` 切换模型内部模块的行为，比如关闭 dropout；`torch.no_grad()` 关闭梯度追踪，让 PyTorch 不记录计算图，也不保存反向传播需要的中间结果。推理时通常两者一起使用。

### Q: best checkpoint 和 final checkpoint 有什么区别？

A: final checkpoint 是训练结束时保存的最后一步模型；best checkpoint 是训练过程中 validation loss 最低时保存的模型。最后一步模型不一定最好，因为它可能已经过拟合训练集。

### Q: early stopping 的 patience 是什么？

A: patience 表示允许 validation loss 连续多少次评估没有改善。如果连续没有改善的次数达到 patience，就提前停止训练，避免继续过拟合或浪费计算。

## 本周最小交付物

- 能解释 `tensor_autograd.py`
- 能跑通 `text_classifier.py`
- 能解释 `Embedding -> pooling -> Linear -> cross_entropy`
- 能在 `experiments/` 里记录一次学习率或 batch size 改动实验
- 能用自己的话回答 5 个 PyTorch 训练循环问题
- 能解释 MiniGPT 的训练流程：`corpus -> stoi/itos -> get_batch -> logits -> cross_entropy -> backward -> step`
- 能解释 MiniGPT 的采样流程：`prompt -> logits[:, -1, :] -> decoding strategy -> cat`
- 能用实验说明 train loss / validation loss 和过拟合信号
- 能比较 `temperature / top-k / top-p / seed / greedy`
- 能解释推理时为什么使用 `model.eval()` 和 `torch.no_grad()`
- 能解释 best checkpoint 和 early stopping 如何用 validation loss 控制训练

## 链接

- [[text-classifier-day2]]
- [[tiny-lm-day3]]
- [[2026-05-06]]
- [[2026-05-07]]
- [[2026-05-08]]
- [[2026-05-09]]
- [[experiments/tiny-lm-block-dim|Tiny LM block/dim 实验]]
- [[lora-qlora]]
- [[transformer-from-scratch]]
- [[llm-inference-systems]]
