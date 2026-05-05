# Tiny Language Model Day 3

## 今天跑了什么

脚本：

```powershell
F:\codex_workspace\LLM\.venv\Scripts\python.exe F:\codex_workspace\LLM\projects\pytorch-llm-basics\tiny_lm.py
```

目标：理解一个最小 next-token language model 是怎么训练和生成文本的。

核心结论：

```text
文本分类器是在问：这句话属于哪一类？
语言模型是在问：下一个 token 是谁？
```

## Corpus 是什么

`CORPUS` 是模型学习用的原始文本材料，可以理解成这个小模型的课本。

在 `tiny_lm.py` 里，模型只从这一小段文本里学习：

```python
CORPUS = (
    "大语言模型 从 token 中 学习 模式。"
    "小实验 能 让 训练 过程 更 容易 理解。"
)
```

真实 LLM 的 corpus 可能是海量网页、书籍、代码、论文和问答数据；这个脚本把过程缩小到显微镜下，方便看清语言模型的基本机制。

## 词表：chars / stoi / itos

```python
torch.manual_seed(42)
chars = sorted(set(CORPUS))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
```

- `torch.manual_seed(42)`：固定随机种子，让实验更容易复现。
- `chars = sorted(set(CORPUS))`：从语料里取出所有出现过的字符，去重并排序。
- `stoi`：string to integer，把字符映射成 token id。
- `itos`：integer to string，把 token id 映射回字符。

可以记成：

```text
文字 -> stoi -> token id -> 模型
模型输出 token id -> itos -> 文字
```

## x 和 y 为什么错开一位

训练语言模型时，`x` 是输入片段，`y` 是标准答案片段。`y` 比 `x` 整体往后平移一位。

例如：

```text
x: 大 语 言 模 型
y: 语 言 模 型 从
```

含义是：

```text
看到“大”，答案是“语”
看到“语”，答案是“言”
看到“言”，答案是“模”
看到“模”，答案是“型”
```

所以用户自己的总结是对的：

```text
因为语言模型的训练目标是给出前文预测下一个字，所以 x 就是前文，y 是往后一位的标准答案。
```

更精确一点：

```text
y 不是模型预测出来的文本，而是拿来和模型预测结果对比的标准答案。
```

## 随机切 batch

```python
for step in range(2000):
    starts = torch.randint(0, len(data) - block_size - 1, (16,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
```

这段代码每一步都做三件事：

- 从整段 `data` 里随机选 16 个起点。
- 每个起点切一段长度为 `block_size` 的输入 `x`。
- 再切一段向后错一位的目标 `y`。

如果 `block_size = 16`，那么：

```text
x.shape = (16, 16)
y.shape = (16, 16)
```

前一个 `16` 是 batch size，表示一轮同时训练 16 条片段；后一个 `16` 是每条片段的 token 数。

## logits 和 cross entropy

模型输出：

```text
logits.shape = (batch_size, block_size, vocab_size)
```

三个维度分别是：

- `batch_size`：这一轮有多少条训练片段。
- `block_size`：每条片段有多少个 token 位置。
- `vocab_size`：每个位置都要对整个词表里的所有 token 打分。

注意：`logits` 是原始分数，不是概率。经过 `softmax` 之后才是概率。

loss 计算：

```python
loss = nn.functional.cross_entropy(logits.view(-1, len(chars)), y.view(-1))
```

为什么要压平：

```text
前两维负责数题目，最后一维负责列选项。
```

原来：

```text
logits: (batch_size, block_size, vocab_size)
y:      (batch_size, block_size)
```

压平后：

```text
logits: (batch_size * block_size, vocab_size)
y:      (batch_size * block_size)
```

这样每一个位置都变成一道独立的“预测下一个 token”的分类题。

## TinyBigramLM 训练时改了什么

模型结构：

```python
self.embedding = nn.Embedding(vocab_size, dim)
self.head = nn.Linear(dim, vocab_size)
```

训练时修改的是模型参数：

```text
embedding.weight
head.weight
head.bias
```

- `embedding.weight`：学习每个字符/token 应该对应什么向量。
- `head.weight` 和 `head.bias`：学习如何根据当前位置的向量，给词表里的下一个 token 打分。

训练不会修改 `CORPUS`，也不会修改 `x` 或 `y`。

## 生成文本过程

```python
for _ in range(80):
    logits = model(context[:, -block_size:])
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    generated.append(itos[next_id.item()])
    context = torch.cat([context, next_id], dim=1)
```

生成过程：

- 看当前上下文。
- 取最后一个位置的 `logits`。
- 用 `softmax` 把分数变成概率。
- 用 `multinomial` 按概率抽样下一个 token。
- 把生成的 token 接回上下文。
- 重复 80 次。

为什么不用永远选最高分：

```text
argmax 是每次选第一名；
multinomial 是按概率加权抽签。
```

抽样会保留随机性和多样性，但也可能让生成文本更不稳定。

## block_size 实验

### block_size = 32

观察：

```text
loss 从约 3.6463 降到约 0.7345
生成文本出现“大语言模式。小实验...”等接近语料的片段
```

结论：模型学到了局部接龙关系，但生成仍然重复、语义不稳定。

### block_size = 64

报错：

```text
RuntimeError: random_ expects 'from' to be less than 'to', but got from=0 >= to=-21
```

原因：

```python
torch.randint(0, len(data) - block_size - 1, (16,))
```

当 `block_size` 太大时，`len(data) - block_size - 1` 变成负数，说明语料长度不够切出这么长的片段，还要再留一位给 `y`。

结论：

```text
block_size 必须小于 len(data) - 1。
```

### block_size = 16

观察：

```text
loss 仍然在 0.7 左右波动
生成结果和 block_size = 32 接近
```

关键洞察：

```text
block_size 变大了，但 TinyBigramLM 不一定变聪明。
```

因为当前模型的 forward 是：

```python
return self.head(self.embedding(input_ids))
```

它没有 attention，也没有循环结构。每个位置主要只根据自己的 token embedding 去预测下一个 token，并没有真正综合前文上下文。

## dim 实验

实验：

```text
batch_size = 16
dim = 128
```

观察：

```text
loss 从约 3.5949 降到约 0.6491
```

相比之前 `dim = 64` 时常见的 `0.73 ~ 0.75`，`dim = 128` 的 loss 略低。

结论：

```text
把 dim 从 64 增加到 128 后，模型参数量变大，loss 有所下降，说明模型容量增加有一定帮助。但生成文本仍然重复、语义不稳定，因为 TinyBigramLM 没有 attention，只能主要学习当前 token 到下一个 token 的局部关系。
```

## 和 text_classifier.py 的区别

用户自己的总结：

```text
tiny_lm.py 输入是一段随机开始、长度为 block_size 的文本，训练目标是预测下一个字是什么。
text_classifier.py 输入是一段长度固定的文本，输出是 label，训练目标是打对 label。
```

更精确版本：

```text
tiny_lm.py 的输入是从语料里随机截取的长度为 block_size 的 token 片段，目标 y 是这个片段整体向后错开一位。模型输出不是直接输出文本，而是每个位置对词表中所有 token 的 logits，训练目标是预测每个位置的下一个 token。

text_classifier.py 的输入是一条被 pad 到固定长度的文本，模型把整句话压成一个向量后输出两个类别的 logits，训练目标是预测这句话的 label。
```

## 今天掌握的概念

- `corpus`：模型学习用的原始文本材料。
- `stoi / itos`：字符和 token id 的双向映射。
- `x / y` 错位：语言模型训练 next-token prediction 的关键。
- `logits`：模型对每个候选 token 的原始分数。
- `cross_entropy`：把每个位置都当成一道分类题来计算损失。
- `softmax + multinomial`：把分数变成概率，再按概率抽样生成。
- `block_size`：每次训练片段的长度，不等于模型一定能理解这么长的上下文。
- `dim`：embedding 向量维度，增大后模型容量变大，但结构限制仍然存在。

## 今日一句话

```text
TinyBigramLM 让我看清了语言模型的最小骨架：把文本变成 token id，用 embedding 表示 token，用 linear 给下一个 token 打分，再用 cross entropy 逼模型把正确答案的分数抬高。
```
