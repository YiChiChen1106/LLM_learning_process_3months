# LLM LoRA 微调

第 5-7 周交付物。

## 目标

用 LoRA 或 QLoRA 微调一个 7B 级开源模型，并产出可复现的评测报告。

## 推荐起步模型

选择一个你能合法下载和运行的模型：

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`，适合更快 dry run
- 其他符合使用要求的开源模型

## 数据规则

只使用公开数据或脱敏样例。不要提交公司数据。

starter 脚本期望 JSONL 格式：

```json
{"instruction": "用一段话解释 LoRA。", "response": "LoRA 会冻结基础模型权重..."}
```

## 命令

在仓库根目录安装依赖：

```bash
pip install -r requirements.txt
```

创建一个安全的 tiny smoke 数据集：

```bash
python prepare_sample_data.py
```

先用小模型做 smoke 微调：

```bash
python train_lora.py --config configs/smoke.yaml
```

## 验收标准

- 训练前保存 baseline 输出。
- LoRA 或 QLoRA adapter 能成功训练。
- 评测能对比 baseline 和微调后的输出。
- README 记录模型、数据集、GPU 配置、精度、训练时间和失败案例。
