from __future__ import annotations

import torch
from torch import nn


CORPUS = (
    "大语言模型 从 token 中 学习 模式。"
    "小实验 能 让 训练 过程 更 容易 理解。"
)


class TinyBigramLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128) -> None:
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

    model = TinyBigramLM(len(chars))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    block_size = 16
    for step in range(2000):
        starts = torch.randint(0, len(data) - block_size - 1, (16,))
        x = torch.stack([data[i : i + block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])

        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, len(chars)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 40 == 0 or step == 1999:
            print(f"步骤={step:03d} 损失={loss.item():.4f}")

    context = torch.tensor([[stoi["大"]]], dtype=torch.long)
    generated = ["大"]
    for _ in range(80):
        logits = model(context[:, -block_size:])
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated.append(itos[next_id.item()])
        context = torch.cat([context, next_id], dim=1)

    print("".join(generated))


if __name__ == "__main__":
    main()
