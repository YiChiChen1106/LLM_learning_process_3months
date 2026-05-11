from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive")

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        for parameter in self.linear.parameters():
            parameter.requires_grad = False

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r

        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def main() -> None:
    torch.manual_seed(0)

    in_features = 128
    out_features = 384
    rank = 8
    alpha = 16
    batch_size = 32

    full_linear = nn.Linear(in_features, out_features, bias=True)
    lora_linear = LoRALinear(in_features, out_features, r=rank, alpha=alpha, bias=True)
    x = torch.randn(batch_size, in_features)

    full_total, full_trainable = count_parameters(full_linear)
    lora_total, lora_trainable = count_parameters(lora_linear)
    lora_out = lora_linear(x)
    max_initial_diff = (lora_out - lora_linear.linear(x)).abs().max().item()

    print("Full fine-tuning Linear(128, 384, bias=True)")
    print(f"  total parameters:     {full_total}")
    print(f"  trainable parameters: {full_trainable}")
    print()
    print("LoRALinear(128, 384, r=8, alpha=16, bias=True)")
    print(f"  total parameters:     {lora_total}")
    print(f"  trainable parameters: {lora_trainable}")
    print(f"  output shape:         {tuple(lora_out.shape)}")
    print(f"  initial max diff vs base Linear: {max_initial_diff}")


if __name__ == "__main__":
    main()
