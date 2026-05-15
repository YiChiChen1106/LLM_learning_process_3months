from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lora_checkpoint import load_lora_adapter
from lora_minigpt import replace_qkv_with_lora
from model import MiniGPT


def apply_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def load_torch_checkpoint(path: str | Path, map_location: str) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_sampling_model(
    checkpoint_path: str | Path,
    device: str,
    lora_adapter: str | Path | None = None,
    lora_rank: int = 8,
    lora_alpha: float = 16,
) -> tuple[MiniGPT, dict[str, int], dict[int, str]]:
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    config = checkpoint["config"]

    model = MiniGPT(**config)
    model.load_state_dict(checkpoint["model"])
    if lora_adapter is not None:
        model = replace_qkv_with_lora(model, r=lora_rank, alpha=lora_alpha)
        load_lora_adapter(model, lora_adapter)

    model.to(device)
    model.eval()
    return model, stoi, itos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--prompt", default="large")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos = load_sampling_model(
        checkpoint_path=args.checkpoint,
        device=device,
        lora_adapter=args.lora_adapter,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    ids = [stoi.get(ch, 0) for ch in args.prompt]
    context = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(args.tokens):
            cropped = context[:, -model.block_size :]
            logits = model(cropped)
            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits / args.temperature
            if args.top_k is not None:
                values, _ = torch.topk(next_token_logits, args.top_k)
                min_value = values[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < min_value,
                    float("-inf"),
                )
            if args.top_p is not None:
                next_token_logits = apply_top_p_filtering(next_token_logits, args.top_p)
            probs = torch.softmax(next_token_logits, dim=-1)
            if args.greedy:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_id], dim=1)

    print("".join(itos[int(i)] for i in context[0].tolist()))


if __name__ == "__main__":
    main()
