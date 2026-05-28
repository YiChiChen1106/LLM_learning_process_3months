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

## 接进 MiniGPT 的 qkv

MiniGPT 的 attention 里有：

```python
self.qkv = nn.Linear(dim, 3 * dim)
```

当 `dim=128` 时：

```text
qkv = Linear(128, 384, bias=True)
```

这里的 `384` 是 `3 * 128`，对应 query、key、value 三组向量。

Day11 的最小实验只替换：

```text
blocks[*].attn.qkv
```

不替换：

```text
attn.proj
ffn
head
```

这样能把学习重点固定在 qkv 的 LoRA 参数流上。

参数量：

```text
单个 qkv full fine-tuning:
384 * 128 + 384 = 49536

单个 qkv LoRA, r=8:
A: 8 * 128 = 1024
B: 384 * 8 = 3072
trainable = 4096

4 个 TransformerBlock:
full qkv = 49536 * 4 = 198144
LoRA trainable = 4096 * 4 = 16384
```

运行结果：

```text
base total parameters:      808242
base trainable parameters:  808242
LoRA total parameters:      824626
LoRA trainable parameters:  16384
output shape:               (2, 8, 50)
```

关键理解：

- LoRA 接进 qkv 后，qkv 的外部输入输出 shape 不变。
- 原始 qkv 权重仍然参与 forward，但被冻结。
- 只有 LoRA 的 `A/B` 会被 optimizer 更新。

## MiniGPT LoRA 训练链路

Day12 的目标不是训练出更好的生成效果，而是确认 LoRA adapter 已经进入真实训练链路：

```python
trainable_params = [
    p
    for p in model.parameters()
    if p.requires_grad
]
optimizer = torch.optim.AdamW(trainable_params, lr=3e-4)
```

这表示：

```text
frozen base 参数：参与 forward，但不累积梯度、不被 optimizer 更新
LoRA A/B 参数：参与 forward，也被 optimizer 更新
```

如果 `logits` 和 `targets` 的 shape 是：

```text
logits:  (batch, seq_len, vocab_size)
targets: (batch, seq_len)
```

那么计算 next-token loss 时要展平为：

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
)
```

例如：

```text
logits:  (2, 8, 50) -> (16, 50)
targets: (2, 8)     -> (16,)
```

Day12 的测试重点：

- optimizer 参数量等于 `requires_grad=True` 的参数量。
- backward 后只有 LoRA 参数有 `.grad`。
- 训练一步后，base qkv 权重不变。
- 训练一步后，`lora_B.weight` 改变。

## MiniGPT LoRA Adapter 保存与加载

`model.state_dict()` 会返回模型里的所有参数和 buffer，不只返回可训练参数。接入 LoRA 后，qkv 里会同时出现 base linear 和 LoRA adapter：

```text
blocks.0.attn.qkv.linear.weight
blocks.0.attn.qkv.linear.bias
blocks.0.attn.qkv.lora_A.weight
blocks.0.attn.qkv.lora_B.weight
```

如果只想保存 adapter，就按 key 名筛选：

```python
lora_state = {
    name: tensor
    for name, tensor in model.state_dict().items()
    if "lora_" in name
}
```

这和训练时的筛选不同：

```text
训练 optimizer 参数：看 parameter.requires_grad
保存 adapter checkpoint：看 state_dict key name
```

保存：

```python
torch.save(lora_state, path)
```

加载：

```python
model = MiniGPT(...)
model = replace_qkv_with_lora(model, r=8, alpha=16)
model.load_state_dict(lora_state, strict=False)
```

必须先接入 LoRA，因为普通 MiniGPT 没有 `lora_A/lora_B` 这些参数位置。`strict=False` 允许 checkpoint 只包含部分参数，但不会自动创建 LoRA 层，也不会自动忽略 shape 不匹配。

Day13 的测试重点：

- adapter checkpoint 只包含 `lora_A/lora_B`。
- adapter checkpoint 参数量是 `16384`。
- 保存后加载到同结构模型，LoRA 权重一致。
- 同一个 base model 加载同一个 adapter 后，logits 一致。

## MiniGPT LoRA Adapter 采样推理

LoRA adapter 不能单独用于 generate，因为它只保存增量矩阵 `lora_A/lora_B`，不包含完整的 embedding、Transformer block、LayerNorm 和 head。

采样时需要：

```text
base checkpoint + LoRA adapter checkpoint
```

正确加载顺序：

```python
checkpoint = torch.load(base_checkpoint)
model = MiniGPT(**checkpoint["config"])
model.load_state_dict(checkpoint["model"])
model = replace_qkv_with_lora(model, r=8, alpha=16)
load_lora_adapter(model, adapter_path)
model.eval()
```

`sample.py` 支持可选 adapter：

```powershell
python sample.py --checkpoint runs/mini_gpt_best.pt --lora-adapter runs/adapter.pt --lora-rank 8 --lora-alpha 16
```

如果不传 `--lora-adapter`，仍然按普通 MiniGPT full checkpoint 采样。
如果传 adapter，`sample.py` 会先加载 base checkpoint，再接入 LoRA 结构，最后加载 adapter 权重。

需要注意：

- adapter 训练时的 `r` 必须和推理时的 `--lora-rank` 一致。
- `strict=False` 允许只加载部分 key，但不能解决 shape 不匹配。
- 同一个 base checkpoint + 同一个 adapter 应该得到一致 logits。

## MiniGPT LoRA Merge 推理

LoRA merge 是把 adapter 产生的增量权重直接合进 base weight。对 PyTorch 的 `nn.Linear` 来说，`weight` shape 是 `(out_features, in_features)`。

以 `qkv = nn.Linear(128, 384)`、`r=8` 为例：

```text
base.weight:        (384, 128)
lora_A.weight:      (8, 128)
lora_B.weight:      (384, 8)
lora_B @ lora_A:    (384, 128)
```

所以 merge 公式是：

```python
delta_weight = lora_B.weight @ lora_A.weight
merged_weight = base.weight + scaling * delta_weight
```

不是 `lora_A.weight @ lora_B.weight`，因为 shape 对不上，而且 forward 顺序是先过 A、再过 B。

merge 前：

```text
block.attn.qkv = LoRALinear(...)
state_dict 里有 lora_A/lora_B
```

merge 后：

```text
block.attn.qkv = nn.Linear(...)
state_dict 里不再有 lora_A/lora_B
```

merge 前后的输出理论上等价，测试里用 `torch.allclose` 检查即可。

`sample.py` 支持内存 merge：

```powershell
python sample.py --checkpoint runs/mini_gpt_best.pt --lora-adapter runs/adapter.pt --merge-lora
```

也可以保存 merged checkpoint：

```powershell
python sample.py --checkpoint runs/mini_gpt_best.pt --lora-adapter runs/adapter.pt --save-merged-checkpoint runs/merged.pt
```

保存出的 `merged.pt` 是完整普通 MiniGPT checkpoint，之后可以直接采样：

```powershell
python sample.py --checkpoint runs/merged.pt --prompt large --greedy
```

## 为什么重要

- 可训练参数量大幅下降。
- 优化器状态占用的显存更少。
- 任务 adapter 更容易保存和切换。
- 对消费级 GPU 更友好，能实践 7B 级模型微调。

## QLoRA 的区别

QLoRA 会以量化形式加载基础模型，常见是 4-bit，同时只训练 LoRA adapter。这样可以显著降低显存需求，让更大的模型能在有限 GPU 上微调。

更具体地说：

- `base model` 是 frozen 的，所以适合量化存储。
- `LoRA adapter` 参数很小，而且要训练，所以通常保留 fp16 / bf16。
- 省下来的主要是 base weight 的存储显存，不是 adapter 的参数量。

## Toy QuantizedLinear

最小 toy 版本可以记成：

```python
q = round(x / scale)
x_hat = q * scale
```

其中：

- `scale = max_abs / 127`
- `qweight` 用 `int8` 存
- `qweight` 和 `scale` 适合放进 `buffer`
- 需要训练的 LoRA 权重才放进 `Parameter`

所以 toy QLoRA 的结构可以看成：

```text
quantized base linear (buffer)
+ LoRA adapter (Parameter)
```

最小 `QLoRALinear` 的 forward 是：

```python
base_out = quantized_base(x)
lora_out = lora_B(lora_A(x)) * scaling
out = base_out + lora_out
```

初始化时如果 `lora_B.weight = 0`，输出就等于 `quantized_base(x)`。训练一步后，`qweight` 不会变，`lora_B.weight` 会变。

QLoRA merge 的 toy 公式是：

```python
base_weight_hat = qweight.float() * scale
delta_weight = lora_B.weight @ lora_A.weight
merged_weight = base_weight_hat + scaling * delta_weight
```

merge 后会变成普通 `nn.Linear`，`state_dict()` 里只剩 `weight/bias`，不再有 `qweight/scale/lora_A/lora_B`。

## 实验想法

- 比较 `r=8`、`r=16`、`r=32`。
- 比较只训练 attention 模块和同时训练 attention + MLP。
- 如果环境支持，比较普通 LoRA 和 4-bit QLoRA。

## 常见误区

- LoRA 冻结 `W`，不代表 forward 不使用 `W`。
- LoRA 通常会让 total parameters 略微增加，因为它额外加了 adapter。
- LoRA 真正大幅减少的是 trainable parameters、梯度和优化器状态。
- 计算 full fine-tuning 参数量时，如果 `bias=True`，不要漏掉 bias。
- LoRA 内部用了 rank 瓶颈，不代表外部输出 shape 会变。
- frozen 参数仍然可以参与 forward，冻结不是从计算路径里删除。
- `targets.view(-1)` 是一维 shape，例如 `(16,)`，不是 `(16, 1)`。
- `state_dict()` 不只包含 trainable 参数，也包含 frozen 参数和 buffer。
- `strict=False` 允许只加载部分 key，但不会自动创建缺失的 LoRA 层。
- LoRA adapter 不能单独推理，必须配同结构 base model。
- adapter 的 rank 和加载时创建的 LoRA rank 必须一致。
- LoRA merge 后参数量会回到普通模型，因为 `lora_A/lora_B` 已经被合进 base weight。
- merged checkpoint 是完整普通 checkpoint，不是 adapter checkpoint。

## MiniGPT LoRA 正式训练

Day16 开始从 smoke training 进入正式 adapter 训练。核心目标不是再训练整个 MiniGPT，而是在已经训练好的 base checkpoint 上，只训练 qkv 的 LoRA adapter。

正确构建顺序：

```python
checkpoint = torch.load(base_checkpoint, map_location="cpu")
model = MiniGPT(**checkpoint["config"])
model.load_state_dict(checkpoint["model"])
model = replace_qkv_with_lora(model, r=8, alpha=16)
```

这里必须先加载 base，再替换 LoRA。原因是 base checkpoint 里的 key 是：

```text
blocks.0.attn.qkv.weight
blocks.0.attn.qkv.bias
```

替换 LoRA 后模型里的 key 会变成：

```text
blocks.0.attn.qkv.linear.weight
blocks.0.attn.qkv.linear.bias
blocks.0.attn.qkv.lora_A.weight
blocks.0.attn.qkv.lora_B.weight
```

训练时 optimizer 只拿可训练参数：

```python
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-4,
)
```

对于当前 MiniGPT 配置，`dim=128, r=8, num_layers=4`：

```text
single qkv LoRA = 8 * 128 + 384 * 8 = 4096
4 blocks = 4096 * 4 = 16384 trainable parameters
```

保存 best adapter 时只保存 `lora_` 参数，不保存完整模型：

```python
lora_state = {
    name: tensor
    for name, tensor in model.state_dict().items()
    if "lora_" in name
}
torch.save(lora_state, adapter_path)
```

这个 adapter 后续必须搭配同一个 base checkpoint 使用，或者先 merge 成普通 checkpoint 再单独采样。

## LoRA 推理路径对比

Day17 验证了三条推理路径：

```text
base checkpoint
base checkpoint + LoRA adapter
merged checkpoint
```

base checkpoint 是完整 MiniGPT，可以直接 sample。LoRA adapter 只保存 `lora_A/lora_B`，必须搭配同一个 base checkpoint 使用。merged checkpoint 已经把 LoRA 增量合进普通 Linear，因此也可以直接 sample。

merge 的核心公式：

```python
delta_weight = lora_B.weight @ lora_A.weight
merged_weight = base_weight + scaling * delta_weight
```

对于 qkv：

```text
lora_B.weight:   (384, 8)
lora_A.weight:   (8, 128)
delta_weight:    (384, 128)
base.weight:     (384, 128)
```

验证 merge 是否正确时，优先比较同一个 input 下的 logits，而不是只看生成文本。文本会受到 sampling 策略影响；logits 是模型 forward 的直接输出。

## Day18 总复盘

可以把 LoRA 的三种形态记成一张表：

| 形态 | 训练什么 | 能否单独 sample | 保存什么 |
| --- | --- | --- | --- |
| full fine-tuning | 全部参数 | 可以 | 完整模型 |
| LoRA adapter | `lora_A/lora_B` | 不可以 | 只有增量 |
| merged checkpoint | 不再训练 | 可以 | 完整模型 |

今天收住的关键结论：

- LoRA 真正省的是 `trainable parameters`，不是 `total parameters`。
- 冻结 base 不等于不参与 forward。
- `base + adapter` 和 `merged checkpoint` 本质上是同一条计算路径的两种写法。
- 验证 merge 等价时，优先看 logits，不只看生成文本。

如果把 LoRA 记成一句话：

```text
冻结大模型，训练小补丁；merge 后补丁并回主模型，直接推理。
```

## 面试问题

- LoRA 为什么省显存？
- LoRA rank 控制什么？
- 量化为什么可能让训练不稳定？
- gradient accumulation 和 effective batch size 是什么关系？
- LoRA 的 `total parameters` 和 `trainable parameters` 有什么区别？
- 为什么 MiniGPT 的 `qkv` 适合作为第一个 LoRA 替换目标？
- LoRA 训练时 optimizer 里应该放哪些参数？
- 为什么第一步训练更适合检查 `lora_B.weight` 是否变化？
- LoRA adapter checkpoint 里应该保存哪些权重？
- 为什么加载 LoRA adapter 前要先把目标 Linear 替换成 LoRALinear？
- `state_dict()`、`requires_grad` 和 optimizer state 分别负责什么？
- LoRA adapter 为什么不能单独 generate？
- sampling 时加载 base checkpoint 和 adapter checkpoint 的顺序是什么？
- 如果 adapter 的 rank 和推理时的 rank 不一致，会发生什么？
- LoRA merge 的公式是什么？
- merge 后为什么 `state_dict()` 里不再有 `lora_A/lora_B`？
- merged checkpoint 和 adapter checkpoint 有什么区别？
- 为什么验证 merged checkpoint 等价时更适合比较 logits，而不是比较生成文本？
