from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="results/transformers_benchmark.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    prompt = "请解释 KV cache 如何提升自回归解码效率。 " * max(1, args.prompt_tokens // 10)
    prompts = [prompt for _ in range(args.batch_size)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.prompt_tokens)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    input_tokens = int(inputs["input_ids"].numel())
    output_tokens = int(outputs.numel())
    generated_tokens = output_tokens - input_tokens
    elapsed = end - start
    tokens_per_second = generated_tokens / elapsed if elapsed > 0 else 0.0

    row = {
        "model": args.model,
        "backend": "transformers",
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "latency_s": round(elapsed, 4),
        "tokens_per_second": round(tokens_per_second, 2),
        "peak_memory_mb": round(gpu_memory_mb(), 2),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    exists = out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
