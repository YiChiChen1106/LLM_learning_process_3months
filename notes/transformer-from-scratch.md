# 从零实现 Transformer 笔记

## Decoder-Only Block

一个 decoder-only Transformer block 包含：

1. token embedding 和 positional embedding；
2. masked multi-head self-attention；
3. 残差连接；
4. LayerNorm；
5. feed-forward network；
6. 另一个残差连接。

## Causal Mask

Causal mask 会阻止当前位置看到未来 token。没有它，语言模型训练时就会泄露答案。

## 训练目标

给定 token：

```text
x_0, x_1, ..., x_n
```

模型在每个位置预测下一个 token：

```text
x_1, x_2, ..., x_{n+1}
```

损失函数是对词表 logits 计算 cross-entropy。

## 面试问题

- 为什么需要位置信息？
- 为什么 self-attention 对序列长度是平方复杂度？
- Q、K、V 的 shape 是什么？
- 训练和自回归生成有什么区别？
