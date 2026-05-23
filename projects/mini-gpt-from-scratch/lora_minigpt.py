from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive")

        self.linear = linear
        for parameter in self.linear.parameters():
            parameter.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features
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


def freeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def replace_qkv_with_lora(model: nn.Module, r: int = 8, alpha: float = 16) -> nn.Module:
    freeze_all_parameters(model)
    for block in model.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, r=r, alpha=alpha)
    return model


def merge_lora_linear(layer: LoRALinear) -> nn.Linear:
    base = layer.linear
    merged = nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
    ).to(device=base.weight.device, dtype=base.weight.dtype)

    delta_weight = layer.lora_B.weight @ layer.lora_A.weight
    with torch.no_grad():
        merged.weight.copy_(base.weight + layer.scaling * delta_weight)
        merged.weight.requires_grad = base.weight.requires_grad
        if base.bias is not None:
            merged.bias.copy_(base.bias)
            merged.bias.requires_grad = base.bias.requires_grad

    return merged


def merge_qkv_lora(model: nn.Module) -> nn.Module:
    for block in model.blocks:
        if isinstance(block.attn.qkv, LoRALinear):
            block.attn.qkv = merge_lora_linear(block.attn.qkv)
    return model


def main() -> None:
    from model import MiniGPT

    torch.manual_seed(0)
    model = MiniGPT(
        vocab_size=50,
        block_size=16,
        dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.0,
    )
    base_total, base_trainable = count_parameters(model)

    lora_model = replace_qkv_with_lora(model, r=8, alpha=16)
    lora_total, lora_trainable = count_parameters(lora_model)
    input_ids = torch.randint(0, 50, (2, 8))
    logits = lora_model(input_ids)

    print("MiniGPT qkv LoRA smoke check")
    print(f"  base total parameters:      {base_total}")
    print(f"  base trainable parameters:  {base_trainable}")
    print(f"  LoRA total parameters:      {lora_total}")
    print(f"  LoRA trainable parameters:  {lora_trainable}")
    print(f"  output shape:               {tuple(logits.shape)}")
    print("  replaced modules:           blocks[*].attn.qkv")


if __name__ == "__main__":
    main()
