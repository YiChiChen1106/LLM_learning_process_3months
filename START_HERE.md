# 从这里开始

这是第一周的启动清单。

## 第 1 天

1. 创建 Python 环境。

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. 运行最小的 PyTorch 练习。

```bash
cd projects/pytorch-llm-basics
python tensor_autograd.py
```

3. 创建今天的学习日志。

```text
把 templates/daily-log.md 复制到 daily/YYYY-MM-DD.md。
```

4. 写下这三个问题的答案：

- `loss.backward()` 做了什么？
- 为什么需要 `optimizer.zero_grad()`？
- 脚本里出现了哪些 tensor shape？

## 第 1 周目标

到周五，你应该能够：

- 解释一个基础 PyTorch 训练循环；
- 跑通文本分类器；
- 修改学习率和 batch size；
- 在 `experiments/` 里记录一个实验。

## 第一次长时间服务器任务

不要一开始就跑 7B 模型。先证明环境可用：

```bash
cd projects/llm-lora-finetuning
python prepare_sample_data.py
python train_lora.py --config configs/smoke.yaml
```

确认 smoke run 成功后，再切到更大的 Qwen 模型和真实公开数据集。
