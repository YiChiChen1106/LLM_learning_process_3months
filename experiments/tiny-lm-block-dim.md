# Tiny LM block_size / dim 实验

日期：2026-05-05

## Summary

本实验比较 `tiny_lm.py` 中 `block_size` 和 `dim` 对 loss、生成文本和报错情况的影响。

实验结论：

```text
block_size 影响每次训练切片长度，但 TinyBigramLM 没有 token 间信息交互，所以增大 block_size 不等于真正理解更长上下文。
dim 增大后模型容量变大，部分实验 loss 会降低，但不能弥补没有 attention 的结构限制。
block_size=64 会报错，因为当前 CORPUS 太短，无法切出长度为 64 的 x 和后移一位的 y。
```

## Command

标准运行命令：

```powershell
F:\codex_workspace\LLM\.venv\Scripts\python.exe F:\codex_workspace\LLM\projects\pytorch-llm-basics\tiny_lm.py
```

实际批量实验使用等价训练逻辑，并从 `tiny_lm.py` 导入 `CORPUS` 和 `TinyBigramLM`，逐组设置：

```text
dim = 64 / 128
block_size = 16 / 32 / 64
batch_size = 16
steps = 2000
lr = 1e-2
seed = 42
```

## Results

| 实验 | dim | block_size | 初始 loss | 最终 loss | 后 10 次打印区间 | 生成现象 |
|---|---:|---:|---:|---:|---|---|
| 1 | 64 | 16 | 3.6368 | 0.7479 | 0.6725 - 0.7555 | 能生成接近语料的局部片段，但重复明显 |
| 2 | 64 | 32 | 3.6463 | 0.7345 | 0.7306 - 0.7554 | 和 `block_size=16` 非常接近 |
| 3 | 64 | 64 | N/A | N/A | N/A | 报错，语料长度不够切片 |
| 4 | 128 | 16 | 3.5949 | 0.6491 | 0.6491 - 0.8209 | loss 更低一些，但仍然重复、语义不稳 |
| 5 | 128 | 32 | 3.6254 | 0.7323 | 0.7245 - 0.7596 | 生成有局部模式，但没有明显长上下文能力 |

## Generated Samples

### 实验 1：dim=64, block_size=16

```text
大语言模式。小实验 过程 中 过程 训练 学习 学习 中 学习 token 理解 学习 中 中 训练 训练 让 中 过程 过程 能 容易 token 学习 让 能
```

### 实验 2：dim=64, block_size=32

```text
大语言模式。小实验 过程 中 过程 训练 学习 学习 中 学习 token 理解 学习 中 中 训练 训练 让 中 过程 过程 能 容易 token 学习 让 能
```

### 实验 3：dim=64, block_size=64

```text
RuntimeError: random_ expects 'from' to be less than 'to', but got from=0 >= to=-21
```

原因：

```python
starts = torch.randint(0, len(data) - block_size - 1, (16,))
```

当 `block_size=64` 时，当前 `CORPUS` 长度不足，`len(data) - block_size - 1` 变成负数。训练需要切出长度为 `block_size` 的 `x`，还要再往后一位切出 `y`，所以至少要有 `block_size + 1` 的可用长度。

### 实验 4：dim=128, block_size=16

```text
大语言模式。小实验 模型 能 中 训练 从 中 能 让 训练 模式。小实验 能 学习 训练 学习 学习 模型 让 中 模式。小实验 token 训练 能 更 训练
```

### 实验 5：dim=128, block_size=32

```text
大语言模型 过程 从 学习 让 能 学习 学习 容易 token 学习 容易 学习 容易 中 过程 训练 更 让 中 模型 从 让 token 训练 能 更 训练
```

## Interpretation

### block_size 变大为什么没有明显变强

`TinyBigramLM` 的 `forward` 是：

```python
return self.head(self.embedding(input_ids))
```

每个 token 先变成自己的 embedding，然后独立经过 linear head 预测下一个 token。不同位置之间没有 attention、RNN 或其他信息混合。

所以：

```text
block_size 只是一次摆了多少道 next-token 预测题；
它不代表模型真的能综合这么长的上下文。
```

这解释了为什么 `block_size=16` 和 `block_size=32` 的结果非常接近。

### dim 变大为什么有时 loss 更低

`dim` 是 embedding 向量维度。`dim=128` 比 `dim=64` 给每个 token 更大的表示空间，也带来更多参数。

因此模型可能更容易记住当前 token 到下一个 token 的局部接续规律，train loss 会低一些。例如：

```text
dim=64, block_size=16: final loss = 0.7479
dim=128, block_size=16: final loss = 0.6491
```

但生成文本仍然重复，因为结构仍然是 `Embedding -> Linear`，没有真正读取前文的能力。

### loss 后期为什么波动

每一步的训练片段来自随机起点：

```python
starts = torch.randint(...)
```

不同 batch 难度不同，所以 loss 不会严格单调下降。更应该看整体趋势和后期区间，而不是只看某一个 step。

## Decision

这五组实验说明：

```text
TinyBigramLM 已经能展示 next-token prediction 的最小训练流程，但它不是 GPT。
GPT 需要 causal self-attention、positional embedding、多层 Transformer block、残差连接、LayerNorm 和 FFN，才能真正利用上下文。
```

第四天之后可以进入两个方向：

- 继续补 validation loss / overfitting 的概念。
- 准备阅读 `projects/mini-gpt-from-scratch`，重点看 causal attention 怎么让 token 之间发生信息交互。
