from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from model import MiniGPT


TRAIN_CORPUS = (
    "大语言模型 被训练 来预测 下一个 token。"
    "attention 让 token 能读取 有用的 历史上下文。"
    "小模型 很适合 用来学习 训练机制。"
) * 200

VAL_CORPUS = (
    "验证集 用来 检查 模型 是否 只是 记住 训练文本。"
    "如果 train loss 很低 但 val loss 很高 就可能 过拟合。"
) * 20


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    vocab_size: int,
    device: str,
) -> tuple[float, float]:
    model.eval()
    losses = []
    for data in (train_data, val_data):
        split_losses = torch.empty(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            split_losses[i] = loss.item()
        losses.append(split_losses.mean().item())
    model.train()
    return losses[0], losses[1]


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chars = sorted(set(TRAIN_CORPUS + VAL_CORPUS))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    train_data = torch.tensor([stoi[ch] for ch in TRAIN_CORPUS], dtype=torch.long)
    val_data = torch.tensor([stoi[ch] for ch in VAL_CORPUS], dtype=torch.long)

    block_size = 64
    batch_size = 64
    eval_iters = 20

    model = MiniGPT(len(chars), block_size=block_size, dim=128, num_heads=4, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(600):
        x, y = get_batch(train_data, block_size, batch_size=batch_size, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, len(chars)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 599:
            train_loss, val_loss = estimate_loss(
                model,
                train_data,
                val_data,
                block_size,
                batch_size,
                eval_iters,
                len(chars),
                device,
            )
            print(f"步骤={step:04d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

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
