from __future__ import annotations

import json
from pathlib import Path


EXAMPLES = [
    {
        "instruction": "用一段话解释 LoRA。",
        "response": "LoRA 会冻结大语言模型的基础权重，只在选定层中训练小型低秩 adapter 矩阵，从而降低微调所需的显存和参数量。",
    },
    {
        "instruction": "什么是 KV cache？",
        "response": "KV cache 会保存历史 token 的 attention key 和 value，让自回归生成在每一步复用历史计算，而不必重新计算完整前缀。",
    },
    {
        "instruction": "为什么 batch size 会影响吞吐？",
        "response": "更大的 batch 往往能提升 GPU 利用率和每秒生成 token 数，但也可能增加单个请求的等待时间和显存占用。",
    },
]


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "sample_train.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for example in EXAMPLES:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"已写入 {path}")


if __name__ == "__main__":
    main()
