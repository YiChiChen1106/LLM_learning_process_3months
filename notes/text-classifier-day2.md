# 文本分类器学习记录

## 今天跑了什么

脚本：

```powershell
cd F:\codex_workspace\LLM\projects\pytorch-llm-basics
F:\codex_workspace\LLM\.venv\Scripts\python.exe text_classifier.py
```

目标：理解一个最小文本分类器如何把中文句子变成数字、向量、分类分数，并通过训练更新模型参数。

## 程序总流程

```text
中文句子
  -> build_vocab() 建词表
  -> encode() 转成固定长度 token id
  -> DataLoader 组成 batch
  -> Embedding 把 id 变成词向量
  -> mean pooling 得到句子向量
  -> Linear 输出两个分类分数 logits
  -> cross_entropy 计算 loss
  -> backward 计算梯度
  -> optimizer.step() 更新模型参数
  -> 统计 loss 和 accuracy
```

一句话理解：

```text
这个程序让模型看 6 句中文训练样本，学习判断一句话是正面还是负面。
```

## 训练集是什么

训练集就是脚本里的 `TEXTS`：

```python
TEXTS = [
    ("这个 模型 有用 且 可靠", 1),
    ("这次 训练 看起来 很 稳定", 1),
    ("这个 结果 很 强 也 很 有帮助", 1),
    ("这个 实验 失败 得 很 严重", 0),
    ("模型 输出 是 错误 的", 0),
    ("训练 不稳定 而且 很 慢", 0),
]
```

- `1`：正面样本。
- `0`：负面样本。
- 一共只有 6 条样本，所以 accuracy 到 `1.00` 只能说明模型把训练集分对了，不代表真的具备很强泛化能力。

## `build_vocab()` 做什么

`build_vocab()` 是在给训练集中出现过的词分配 ID。

```text
build_vocab() = 造词典
```

例子：

```text
<pad> -> 0
<unk> -> 1
这个 -> 2
模型 -> 3
有用 -> 4
```

神经网络不能直接处理中文字符串，只能处理数字张量，所以要先把词变成 ID。

## `encode()` 做什么

`encode()` 是拿词表把一句话翻译成固定长度的 token id 列表。

```text
encode() = 拿词典翻译句子
```

例子：

```text
这个 模型 有用 且 可靠
```

可能变成：

```text
[2, 3, 4, 5, 6, 0, 0, 0]
```

这里的 `0` 是 `<pad>`，用于把不同长度的句子补齐到同样长度。

注意：`encode()` 不是直接生成向量，它生成的是 token id。真正把 id 变成向量的是后面的 `Embedding`。

## `MeanPoolClassifier` 做什么

这个模型的数据流是：

```text
token id -> token 向量 -> 句子向量 -> 两个分类分数
```

核心模块：

```python
self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
self.classifier = nn.Linear(dim, 2)
```

`Embedding`：

- 输入：token id。
- 输出：每个 token 对应的向量。
- 如果 `input_ids` 形状是 `(2, 8)`，`dim=32`，那么 `embedded` 形状是 `(2, 8, 32)`。

`mean pooling`：

- 把一句话里的多个 token 向量平均成一个句子向量。
- 要用 `mask` 排除 `<pad>`，因为 `<pad>` 不是真实词，参与平均会污染句子向量。

`Linear(dim, 2)`：

- 输出两个 logits。
- 类别 0：负面分数。
- 类别 1：正面分数。
- 谁的分数更大，模型就更倾向预测谁。

## `logits.argmax(dim=-1)` 做什么

`logits` 是模型输出的两个分类分数，不是概率。

例子：

```text
[2.7, 0.5] -> 预测 0，负面
[0.3, 2.1] -> 预测 1，正面
```

`logits.argmax(dim=-1)` 会选出最后一维里最大分数的位置，得到预测标签。

accuracy 的计算就是：

```text
预测标签和真实 label 比较，看对了几个。
```

## 训练时哪些参数在变

训练过程中变化的是模型里的参数，不是原始中文句子，也不是标签。

这个脚本里会更新：

- `embedding` 里的词向量参数。
- `classifier.weight`。
- `classifier.bias`。

因为优化器拿到的是：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
```

所以 `optimizer.step()` 更新的是 `model.parameters()` 里的可训练参数。

## 学习率实验

实验：修改 `lr`，观察 loss 和 accuracy。

### `lr=3e-2`

```text
轮次=00 损失=0.6683 准确率=0.83
轮次=05 损失=0.0253 准确率=1.00
轮次=10 损失=0.0016 准确率=1.00
轮次=29 损失=0.0004 准确率=1.00
```

观察：比默认学习率收敛更快，第 5 轮已经达到 `1.00` accuracy。

### `lr=3e-1`

```text
轮次=00 损失=1.1143 准确率=0.50
轮次=05 损失=0.0000 准确率=1.00
轮次=29 损失=0.0000 准确率=1.00
```

观察：初始 loss 更高，但仍然很快把 6 条训练样本记住。

### `lr=3`

```text
轮次=00 损失=53.6385 准确率=0.33
轮次=05 损失=0.0000 准确率=1.00
轮次=29 损失=0.0000 准确率=1.00
```

观察：第 0 轮 loss 暴涨，说明学习率已经非常激进，训练有不稳定迹象。但因为训练集太小，模型后面仍然能背下训练集。

### 学习率实验结论

```text
在这个极小训练集上，增大学习率会让模型更快拟合训练集；但当 lr 增大到 3 时，初始 loss 暴涨，说明参数更新步子过大。训练 accuracy=1.00 不代表学习率一定好，因为模型可能只是记住了 6 条训练样本。
```

## Batch Size 实验

实验：固定学习率，比较 `batch_size=1`、`2`、`6`。

### `batch_size=1`

```text
轮次=00 损失=0.6783 准确率=0.83
轮次=05 损失=0.5105 准确率=1.00
轮次=29 损失=0.0872 准确率=1.00
```

### `batch_size=2`

```text
轮次=00 损失=0.6714 准确率=0.83
轮次=05 损失=0.5487 准确率=0.83
轮次=10 损失=0.4454 准确率=1.00
轮次=29 损失=0.1661 准确率=1.00
```

### `batch_size=6`

```text
轮次=00 损失=0.6721 准确率=0.83
轮次=05 损失=0.6048 准确率=0.83
轮次=10 损失=0.5423 准确率=1.00
轮次=29 损失=0.3305 准确率=1.00
```

### Batch size 实验结论

训练集一共 6 条样本：

```text
batch_size=1 -> 每个 epoch 更新 6 次参数
batch_size=2 -> 每个 epoch 更新 3 次参数
batch_size=6 -> 每个 epoch 更新 1 次参数
```

所以在相同 epoch 数下：

```text
batch_size 越小，参数更新次数越多，loss 下降越快。
```

这次实验的最终 loss：

```text
batch_size=1 -> 0.0872
batch_size=2 -> 0.1661
batch_size=6 -> 0.3305
```

注意：这不是严格公平比较，因为三组的总 update step 数不同。如果要公平比较优化效果，应该控制总 update step 数一致。

## 今天小测的掌握点

- 文本不能直接喂给模型，要先变成 token id。
- `build_vocab()` 是造词典，`encode()` 是拿词典翻译句子。
- `<pad>` 只是补位，不应该参与 mean pooling。
- `Embedding` 把 token id 变成向量。
- `Linear(dim, 2)` 输出两个类别分数。
- `argmax` 选最大分数对应的类别。
- 训练改变的是模型参数，不是原始文本。

## 面试表达

这个文本分类器先用 `build_vocab()` 给训练集中的词分配 ID，再用 `encode()` 把句子变成固定长度 token id。模型用 `Embedding` 把 token id 变成词向量，通过 mask 排除 padding 后做 mean pooling，得到句子向量，最后用线性层输出正负两个类别的 logits。训练时 `cross_entropy` 计算分类 loss，`backward()` 计算 embedding 和 classifier 参数的梯度，`optimizer.step()` 更新这些模型参数。
