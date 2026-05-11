# LoRA / QLoRA 笔记

## 核心思想

LoRA 冻结基础模型权重，只在选定的线性层上训练小型低秩更新矩阵。

对于预训练权重矩阵 `W`，LoRA 学习：

```text
W' = W + BA * alpha / r
```

其中 `A` 和 `B` 是小型可训练矩阵，`r` 是 rank，`alpha` 用来缩放更新量。

## Toy LoRA 入门

今天先不使用 Hugging Face，只看一个普通 `nn.Linear`。

对于：

```python
nn.Linear(128, 384, bias=True)
```

PyTorch 里的参数 shape 是：

```text
weight: (384, 128)
bias:   (384,)
```

所以 full fine-tuning 的可训练参数量是：

```text
384 * 128 + 384 = 49536
```

LoRA 不直接训练完整的 `W`，而是额外训练两个小矩阵：

```text
A: (r, in_features)
B: (out_features, r)
```

当 `r=8` 时：

```text
A: (8, 128)   => 1024
B: (384, 8)   => 3072
trainable     => 4096
```

这说明 LoRA 的重点是减少可训练参数量，而不是减少 forward 里使用的原始模型参数。

## 最小 LoRALinear

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        for p in self.linear.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out
```

关键点：

- `self.linear` 仍然参与 forward，但不参与训练。
- `lora_A` 和 `lora_B` 参与训练。
- 最终输出是 `base_out + lora_out`。
- 常见初始化是 `A` 随机初始化、`B` 初始化为 0，这样一开始 `lora_out=0`。

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

## 常见误区

- LoRA 冻结 `W`，不代表 forward 不使用 `W`。
- LoRA 通常会让 total parameters 略微增加，因为它额外加了 adapter。
- LoRA 真正大幅减少的是 trainable parameters、梯度和优化器状态。
- 计算 full fine-tuning 参数量时，如果 `bias=True`，不要漏掉 bias。

## 面试问题

- LoRA 为什么省显存？
- LoRA rank 控制什么？
- 量化为什么可能让训练不稳定？
- gradient accumulation 和 effective batch size 是什么关系？
- LoRA 的 `total parameters` 和 `trainable parameters` 有什么区别？
