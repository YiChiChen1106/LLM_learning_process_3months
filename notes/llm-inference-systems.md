# LLM 推理系统笔记

## 核心术语

- Prefill：处理 prompt token，并构建初始 KV cache。
- Decode：利用缓存的 key/value，每次生成一个新 token。
- TTFT：time to first token，首 token 延迟。
- Throughput：吞吐，通常指每秒生成 token 数。
- Latency：延迟，用户等待一次响应的时间。
- KV cache：缓存历史 token 的 attention key 和 value。

## 关键权衡

Batching 通常能提升吞吐，因为 GPU 每次调度能做更多工作；但它也可能增加延迟，因为单个请求要等待其他请求一起组成 batch。

更长上下文会增加 KV cache 显存，近似和下面这些因素成正比：

```text
层数 * 序列长度 * hidden size * 每个数值的字节数
```

量化可以降低显存和带宽压力，但可能带来质量下降或后端限制。

## 实验想法

- 测 batch size 如何影响吞吐和延迟。
- 测 context length 如何影响显存。
- 如果支持，比较 fp16、int8、int4。
- 比较 Transformers generation 和 vLLM。

## 面试问题

- 为什么 decode 阶段经常受显存带宽限制？
- KV cache 里存的是什么？
- vLLM 为什么能提高 serving 效率？
- 优化 latency 和优化 throughput 有什么区别？
