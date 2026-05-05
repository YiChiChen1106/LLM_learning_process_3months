from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/transformers_benchmark.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.rename(
        columns={
            "backend": "后端",
            "model": "模型",
            "batch_size": "batch_size",
            "prompt_tokens": "prompt_tokens",
            "latency_s": "延迟_秒",
            "tokens_per_second": "每秒token数",
            "peak_memory_mb": "峰值显存_MB",
        }
    )
    columns = [
        "后端",
        "模型",
        "batch_size",
        "prompt_tokens",
        "延迟_秒",
        "每秒token数",
        "峰值显存_MB",
    ]
    print(df[columns].sort_values(["后端", "batch_size", "prompt_tokens"]).to_string(index=False))


if __name__ == "__main__":
    main()
