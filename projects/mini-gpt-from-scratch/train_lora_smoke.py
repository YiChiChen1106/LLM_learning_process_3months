import torch
import torch.nn.functional as F

from lora_minigpt import count_parameters, replace_qkv_with_lora
from model import MiniGPT


def main() -> None:
    torch.manual_seed(0)

    model = MiniGPT(
        vocab_size=50,
        block_size=16,
        dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.0,
    )
    model = replace_qkv_with_lora(model, r=8, alpha=16)

    total, trainable = count_parameters(model)
    print(f"total parameters: {total}")
    print(f"trainable parameters: {trainable}")
    trainable_items = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    print("trainable names:")
    for name, _ in trainable_items:
        print(f"  {name}")

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable_items],
        lr=3e-4,
    )

    for step in range(30):
        input_ids = torch.randint(0, 50, (2, 8))
        targets = torch.randint(0, 50, (2, 8))

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step:02d} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
