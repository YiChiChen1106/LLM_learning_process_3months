# 简历 Bullet 素材

这个文件用来积累可量化的简历表达。只有实验完成后，才把占位符替换成真实数字。

## 微调项目

- 使用 PyTorch、Transformers、PEFT 和双 RTX 4090 GPU，对 7B 级开源 LLM 进行 LoRA/QLoRA 领域任务微调，将 `[指标]` 从 `[baseline]` 提升到 `[结果]`。
- 构建可复现的 LLM 微调流程，覆盖数据预处理、adapter 训练、评测、错误分析和实验配置记录。

## 推理 Benchmark 项目

- 在双 RTX 4090 GPU 上对比 `[后端 A]` 和 `[后端 B]` 的 LLM 推理性能，测量 TTFT、tokens/s、GPU 显存、batch size 扩展和 context length 扩展。
- 分析 KV cache 与 batching 的权衡，在 `[约束条件]` 下将吞吐提升 `[x]%`。

## Mini GPT 项目

- 使用 PyTorch 从零实现 decoder-only Transformer 语言模型，包含 causal self-attention、multi-head attention、LayerNorm、训练和采样。

## 规则

- 每条 bullet 最好包含数字、方法和结果。
- 不要提公司私有数据。
- 简历里写出的每一个指标，都要能解释来源和复现实验命令。
