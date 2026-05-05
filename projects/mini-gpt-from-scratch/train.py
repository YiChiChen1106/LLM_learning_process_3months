from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from model import MiniGPT


CORPUS = (
    "大语言模型 被训练 来预测 下一个 token。"
    "attention 让 token 能读取 有用的 历史上下文。"
    "小模型 很适合 用来学习 训练机制。"
) * 200


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts]).to(device)
    return x, y


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chars = sorted(set(CORPUS))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in CORPUS], dtype=torch.long)

    block_size = 64
    model = MiniGPT(len(chars), block_size=block_size, dim=128, num_heads=4, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(600):
        x, y = get_batch(data, block_size, batch_size=64, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, len(chars)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 599:
            print(f"步骤={step:04d} 损失={loss.item():.4f}")

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "config": {"vocab_size": len(chars), "block_size": block_size, "dim": 128, "num_heads": 4, "num_layers": 4},
        },
        out_dir / "mini_gpt.pt",
    )


if __name__ == "__main__":
    main()
