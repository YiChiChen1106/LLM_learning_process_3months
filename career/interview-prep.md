# 面试准备

## PyTorch

问：`loss.backward()` 做了什么？

答：PyTorch 会沿着 forward 时动态构建的计算图反向传播，为 `requires_grad=True` 的 tensor 计算梯度。之后 optimizer 会使用这些梯度更新参数。

问：为什么要调用 `optimizer.zero_grad()`？

答：PyTorch 默认会累积梯度。清空梯度可以避免上一个 batch 的梯度污染当前 batch 的更新。

## Transformer

问：为什么 decoder-only LLM 需要 causal mask？

答：语言模型训练目标是用前面的 token 预测下一个 token。causal mask 可以防止模型看到未来 token。

问：为什么长上下文下 attention 很贵？

答：标准 self-attention 会让每个 token 和其他 token 两两计算关系，所以 attention score 的规模随序列长度平方增长。

## LoRA / QLoRA

问：LoRA 训练的是什么？

答：LoRA 冻结基础模型，只训练小型低秩 adapter 矩阵，用这些矩阵近似选定线性层的权重更新。

问：为什么使用 QLoRA？

答：QLoRA 通过量化方式加载基础模型，同时只训练 LoRA adapter，从而降低显存需求，让更大的模型能在有限 GPU 上微调。

## 推理系统

问：KV cache 是什么？

答：KV cache 缓存历史 token 的 attention key 和 value，使自回归生成时不需要每一步都重新计算完整前缀。

问：TTFT 和 throughput 有什么区别？

答：TTFT 衡量第一个生成 token 多快出现；throughput 衡量系统每秒能生成多少 token，通常用于评估整体处理能力。

## 项目讲解模板

每个项目都按这个结构讲：

1. 问题：要解决什么任务或瓶颈？
2. 约束：模型大小、GPU 显存、时间或数据限制是什么？
3. 方法：你具体构建了什么？
4. 结果：指标和对比是什么？
5. 收获：这个项目改变了你对什么问题的理解？
