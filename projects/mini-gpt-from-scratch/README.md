# 从零实现 Mini GPT

第 3-4 周交付物。

## 目标

用 PyTorch 实现一个小型 decoder-only Transformer，并在小语料上训练。

## 命令

```bash
python train.py
python sample.py --checkpoint runs/mini_gpt.pt --prompt "大模型"
```

## 需要实现的内容

- Token embedding 和 positional embedding。
- Causal multi-head self-attention。
- Feed-forward network。
- 残差连接和 LayerNorm。
- Next-token cross-entropy loss。
- 自回归采样。

## 验收标准

- 模型训练没有 shape 错误。
- toy corpus 上 loss 能下降。
- 采样文本能体现局部模式。
- README 能解释 causal mask 为什么必要。
