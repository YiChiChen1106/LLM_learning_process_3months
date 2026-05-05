# PyTorch LLM 基础

第 1-2 周交付物。

## 目标

建立足够的 PyTorch 熟练度，能看懂并修改 LLM 训练代码。

## 任务

1. 运行 tensor 和 autograd 练习。
2. 训练一个小型文本分类器。
3. 训练一个 tiny next-token language model。
4. 记录 loss 曲线，并解释发生了什么。

## 命令

```bash
python tensor_autograd.py
python text_classifier.py
python tiny_lm.py
```

## 验收标准

- 能解释 `forward`、`backward`、optimizer step 和 gradient accumulation。
- 能解释为什么 train loss 下降时 validation loss 可能变差。
- 能估算模型参数的显存占用。

## 实验记录

实验笔记放到 `../../experiments/`。
