from __future__ import annotations

import argparse

import torch

from model import MiniGPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="large")
    parser.add_argument("--tokens", type=int, default=120)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    config = checkpoint["config"]
    model = MiniGPT(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ids = [stoi.get(ch, 0) for ch in args.prompt]
    context = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(args.tokens):
        cropped = context[:, -model.block_size :]
        logits = model(cropped)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_id], dim=1)

    print("".join(itos[int(i)] for i in context[0].tolist()))


if __name__ == "__main__":
    main()
