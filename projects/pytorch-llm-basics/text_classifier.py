from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TEXTS = [
    ("这个 模型 有用 且 可靠", 1),
    ("这次 训练 看起来 很 稳定", 1),
    ("这个 结果 很 强 也 很 有帮助", 1),
    ("这个 实验 失败 得 很 严重", 0),
    ("模型 输出 是 错误 的", 0),
    ("训练 不稳定 而且 很 慢", 0),
]


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for token in text.split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int = 8) -> list[int]:
    ids = [vocab.get(token, vocab["<unk>"]) for token in text.split()]
    ids = ids[:max_len]
    return ids + [vocab["<pad>"]] * (max_len - len(ids))


class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.classifier = nn.Linear(dim, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != 0).float().unsqueeze(-1)
        embedded = self.embedding(input_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def main() -> None:
    torch.manual_seed(42)
    texts = [item[0] for item in TEXTS]
    labels = torch.tensor([item[1] for item in TEXTS], dtype=torch.long)
    vocab = build_vocab(texts)
    inputs = torch.tensor([encode(text, vocab) for text in texts], dtype=torch.long)

    loader = DataLoader(TensorDataset(inputs, labels), batch_size=6, shuffle=True)
    model = MeanPoolClassifier(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for epoch in range(30):
        total_loss = 0.0
        correct = 0
        for batch_inputs, batch_labels in loader:
            logits = model(batch_inputs)
            loss = nn.functional.cross_entropy(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_inputs.size(0)
            correct += (logits.argmax(dim=-1) == batch_labels).sum().item()

        if epoch % 5 == 0 or epoch == 29:
            print(f"轮次={epoch:02d} 损失={total_loss / len(labels):.4f} 准确率={correct / len(labels):.2f}")


if __name__ == "__main__":
    main()
