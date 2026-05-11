# Toy LoRA From Scratch

Day10 的 PyTorch 手写 LoRA 小实验。

目标：

- 理解普通 `nn.Linear` 的参数量。
- 手写最小 `LoRALinear`。
- 对比 full fine-tuning 和 LoRA 的 `total parameters` / `trainable parameters`。
- 验证 LoRA 冻结原始 Linear，但仍然通过 `base_out + lora_out` 改变输出。

运行：

```powershell
F:\codex_workspace\LLM\.venv\Scripts\python.exe projects\toy-lora-from-scratch\toy_lora_linear.py
```

测试：

```powershell
F:\codex_workspace\LLM\.venv\Scripts\python.exe -m unittest discover -s projects\toy-lora-from-scratch -p test_*.py
```
