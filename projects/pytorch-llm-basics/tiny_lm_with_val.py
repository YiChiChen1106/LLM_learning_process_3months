from __future__ import annotations

import torch
from torch import nn


CORPUS = (
    "大语言模型 从 token 中 学习 模式。"
    "小实验 能 让 训练 过程 更 容易 理解。"
)


class TinyBigramLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(input_ids))


def main() -> None:
    torch.manual_seed(42)

    chars = sorted(set(CORPUS))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in CORPUS], dtype=torch.long)

    block_size = 16
    batch_size = 16
    eval_iters = 20
    max_steps = 2000

    split = int(0.6 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    def check_source(name: str, source: torch.Tensor) -> None:
        min_len = block_size + 1
        if len(source) < min_len:
            raise ValueError(
                f"{name} 太短：长度={len(source)}，至少需要 {min_len}。"
                f"请减小 block_size 或增加 CORPUS。"
            )

    check_source("train_data", train_data)
    check_source("val_data", val_data)

    model = TinyBigramLM(len(chars), dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def get_batch(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(source) - block_size, (batch_size,))
        x = torch.stack([source[i : i + block_size] for i in starts])
        y = torch.stack([source[i + 1 : i + block_size + 1] for i in starts])
        return x, y

    @torch.no_grad()
    def estimate_loss(source: torch.Tensor) -> float:
        model.eval()
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(source)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, len(chars)), y.view(-1))
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    print(f"数据长度={len(data)} train={len(train_data)} val={len(val_data)}")
    print(f"block_size={block_size} batch_size={batch_size} vocab_size={len(chars)}")

    for step in range(max_steps):
        x, y = get_batch(train_data)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, len(chars)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 40 == 0 or step == max_steps - 1:
            train_loss = estimate_loss(train_data)
            val_loss = estimate_loss(val_data)
            print(
                f"步骤={step:04d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
            )

    context = torch.tensor([[stoi[CORPUS[0]]]], dtype=torch.long)
    generated = [CORPUS[0]]
    for _ in range(80):
        logits = model(context[:, -block_size:])
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated.append(itos[next_id.item()])
        context = torch.cat([context, next_id], dim=1)

    print("".join(generated))


if __name__ == "__main__":
    main()
