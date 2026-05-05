# LLM 推理 Benchmark

第 8-10 周交付物。

## 目标

在双 RTX 4090 服务器上 benchmark LLM 推理，并解释吞吐、延迟、显存和 batching 权衡。

## 指标

- TTFT：time to first token，首 token 延迟。
- 端到端延迟。
- 每秒生成 token 数。
- 峰值 GPU 显存。
- batch size 和 context length 扩展表现。

## 命令

Transformers 后端：

```bash
python benchmark_transformers.py --model Qwen/Qwen2.5-0.5B-Instruct --batch-size 1 --max-new-tokens 64
```

结果文件会保存到 `results/`，重要结论整理到 `../../experiments/` 下的实验报告。

## 验收标准

- 至少测试两个 batch size。
- 至少测试两个 context length。
- 如果 vLLM 或 llama.cpp 可用，至少做一次后端对比。
- 写出解释 latency/throughput 权衡的结论。
