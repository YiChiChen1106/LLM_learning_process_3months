# LLM 学习总入口

## 今天从哪里开始

- 当前阶段：第 1-2 周到 LoRA 过渡
- 当前项目：`projects/mini-gpt-from-scratch`
- 今天任务：
  - 把 LoRA 接进 MiniGPT 的 `attn.qkv`
  - 验证 logits shape 不变
  - 统计 MiniGPT LoRA 的 total/trainable 参数量

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
- 学习记录：[[2026-05-06]]、[[2026-05-07]]、[[2026-05-08]]、[[2026-05-09]]、[[2026-05-10]]、[[2026-05-12]]
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
  - `max_steps`、`eval_interval`、`patience`、`learning_rate` 分别控制什么？
  - 为什么 `eval_interval` 不改变参数更新频率？
  - 为什么 `qkv = Linear(dim, 3 * dim)`？
  - 把 LoRA 接进 qkv 后，为什么 attention 后续 shape 不变？

### 5. LoRA / QLoRA

- 目标：理解参数高效微调
- 学习记录：[[2026-05-11]]、[[2026-05-12]]
- Toy 项目：`projects/toy-lora-from-scratch`
- MiniGPT 项目：`projects/mini-gpt-from-scratch`
- 关键问题：
  - 普通 `nn.Linear` 的 `weight` 和 `bias` shape 怎么算？
  - LoRA 为什么只训练少量参数？
  - LoRA 冻结原始 `W` 后，为什么输出仍然能改变？
  - rank 和 alpha 分别影响什么？
  - `total parameters` 和 `trainable parameters` 有什么区别？
  - 只替换 MiniGPT 的 `attn.qkv` 时，哪些参数会训练？
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

### 误区：`eval_interval` 越小，参数更新越频繁

- 错在哪里：参数更新由训练循环里的 `loss.backward()` 和 `optimizer.step()` 决定，仍然每个训练 step 执行。
- 正确理解：`eval_interval` 只控制多久额外评估一次 train/val loss。
- 记忆句：评卷更勤，不代表写卷更多。

### 误区：命令行参数写了就一定生效

- 错在哪里：加了 argparse 参数，但如果后面的训练逻辑仍然使用写死值，参数不会真正影响实验。
- 正确理解：命令行参数要接入实际逻辑，例如 `step % args.eval_interval`。
- 记忆句：旋钮接上线，转动才有用。

### 误区：LoRA 让模型总参数量变少

- 错在哪里：LoRA 会保留原始 Linear，并额外增加低秩矩阵 `A/B`。
- 正确理解：LoRA 主要减少的是可训练参数、梯度和优化器状态，不是 total parameters。
- 记忆句：书还在，只多夹了一张小改错纸；真正重写的只有那张纸。

### 误区：冻结 `W` 就等于 forward 不用 `W`

- 错在哪里：冻结只表示不计算或不更新 `W` 的梯度。
- 正确理解：LoRA forward 仍然计算 `base_out = x @ W.T + b`，再加上 `lora_out`。
- 记忆句：冻结是不改，不是不用。

### 误区：LoRA 内部降到 rank 后，外部输出 shape 也会变

- 错在哪里：LoRA 的 `A` 会把特征降到 rank，但 `B` 会把它升回原始输出维度。
- 正确理解：`LoRALinear` 对外仍然保持和原始 Linear 相同的输入输出 shape。
- 记忆句：中间变窄，出口不变。

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

### Q: `eval_interval` 会不会改变模型每一步的参数更新？

A: 不会。`eval_interval` 只控制隔多少 step 运行一次评估逻辑，比如计算 train loss、validation loss、保存 best checkpoint、检查 early stopping。参数更新仍然发生在每个训练 step。

### Q: `max_steps` 和 early stopping 谁决定训练结束？

A: 两者都可能决定训练结束。`max_steps` 是最多训练多少步的硬上限；early stopping 是当 validation loss 连续多次不改善时提前停止。哪个条件先满足，训练就在哪里结束。

### Q: learning rate 太大可能会带来什么问题？

A: learning rate 太大时，每次参数更新步子过大，可能导致训练不稳定。在当前 toy corpus 实验里，它让 train loss 很快降到很低，但 validation loss 大幅升高，说明模型更快过拟合训练集，泛化更差。

### Q: LoRA 冻结原始 `W`，为什么模型输出仍然可以改变？

A: LoRA 冻结原始 `W`，但额外训练一个低秩更新分支。最终输出是 `base_out + lora_out`，所以即使原始权重不更新，新增的 LoRA 分支也能改变模型行为。

### Q: LoRA 的 `total parameters` 和 `trainable parameters` 有什么区别？

A: `total parameters` 包括原始 Linear 和新增的 LoRA A/B；`trainable parameters` 只包括 `requires_grad=True` 的参数。LoRA 通常会让 total parameters 略微增加，但会大幅减少 trainable parameters。

### Q: MiniGPT 里为什么 `qkv` 是 `Linear(128, 384)`？

A: MiniGPT 的 hidden dim 是 128，attention 需要同时生成 query、key、value 三组向量，所以输出维度是 `3 * dim = 384`。

### Q: 把 LoRA 接进 MiniGPT 的 `qkv` 后，为什么 logits shape 不变？

A: LoRA 内部通过 `A/B` 做低秩更新，但对外仍然保持原始 qkv 的输入输出 shape，也就是 `128 -> 384`。后续 attention、block 和 head 接收到的张量形状没有变，所以 logits 仍然是 `(batch, seq_len, vocab_size)`。

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
- 能用命令行参数控制 MiniGPT 的 `max_steps / eval_interval / patience / learning_rate`
- 能解释普通 `nn.Linear` 和 LoRA 的参数量差异
- 能手写最小 `LoRALinear`
- 能把 LoRA 接进 MiniGPT 的 `attn.qkv` 并解释参数量变化

## 链接

- [[text-classifier-day2]]
- [[tiny-lm-day3]]
- [[2026-05-06]]
- [[2026-05-07]]
- [[2026-05-08]]
- [[2026-05-09]]
- [[2026-05-10]]
- [[2026-05-11]]
- [[2026-05-12]]
- [[experiments/tiny-lm-block-dim|Tiny LM block/dim 实验]]
- [[lora-qlora]]
- [[transformer-from-scratch]]
- [[llm-inference-systems]]
