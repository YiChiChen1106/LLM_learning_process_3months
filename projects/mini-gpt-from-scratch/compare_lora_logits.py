from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from sample import load_sampling_model


@dataclass(frozen=True)
class LogitsCompareResult:
    adapter_logits_shape: tuple[int, ...]
    merged_logits_shape: tuple[int, ...]
    max_abs_diff: float
    allclose: bool


def encode_prompt(prompt: str, stoi: dict[str, int], device: str) -> torch.Tensor:
    ids = [stoi.get(ch, 0) for ch in prompt]
    return torch.tensor([ids], dtype=torch.long, device=device)


def compare_lora_logits(
    checkpoint_path: str | Path,
    lora_adapter_path: str | Path,
    merged_checkpoint_path: str | Path,
    prompt: str = "large",
    lora_rank: int = 8,
    lora_alpha: float = 16,
    atol: float = 1e-6,
    device: str = "cpu",
) -> LogitsCompareResult:
    adapter_model, stoi, _ = load_sampling_model(
        checkpoint_path=checkpoint_path,
        device=device,
        lora_adapter=lora_adapter_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    merged_model, _, _ = load_sampling_model(
        checkpoint_path=merged_checkpoint_path,
        device=device,
    )
    input_ids = encode_prompt(prompt, stoi, device=device)

    with torch.no_grad():
        adapter_logits = adapter_model(input_ids)
        merged_logits = merged_model(input_ids)

    max_abs_diff = float((adapter_logits - merged_logits).abs().max().item())
    return LogitsCompareResult(
        adapter_logits_shape=tuple(adapter_logits.shape),
        merged_logits_shape=tuple(merged_logits.shape),
        max_abs_diff=max_abs_diff,
        allclose=torch.allclose(adapter_logits, merged_logits, atol=atol),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lora-adapter", required=True)
    parser.add_argument("--merged-checkpoint", required=True)
    parser.add_argument("--prompt", default="large")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = compare_lora_logits(
        checkpoint_path=args.checkpoint,
        lora_adapter_path=args.lora_adapter,
        merged_checkpoint_path=args.merged_checkpoint,
        prompt=args.prompt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        atol=args.atol,
        device=args.device,
    )
    print("LoRA logits equivalence check")
    print(f"  adapter logits shape: {result.adapter_logits_shape}")
    print(f"  merged logits shape:  {result.merged_logits_shape}")
    print(f"  max abs diff:         {result.max_abs_diff:.10f}")
    print(f"  allclose:             {result.allclose}")

    if not result.allclose:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
