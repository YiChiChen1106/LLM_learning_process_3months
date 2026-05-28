import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be an nn.Linear")

        weight = linear.weight.detach().to(torch.float32)
        max_abs = weight.abs().max()
        scale_value = (max_abs / 127.0).item() if max_abs.item() > 0 else 1.0 / 127.0
        scale = torch.tensor(scale_value, dtype=torch.float32, device=weight.device)
        qweight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

        if linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

    def dequantize_weight(self) -> torch.Tensor:
        return self.qweight.float() * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize_weight(), self.bias)


class QLoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive")

        self.base = QuantizedLinear(linear)
        self.lora_A = nn.Linear(linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear.out_features, bias=False)
        self.scaling = alpha / r

        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out


def merge_qlora_linear(layer: QLoRALinear) -> nn.Linear:
    if not isinstance(layer, QLoRALinear):
        raise TypeError("layer must be a QLoRALinear")

    has_bias = layer.base.bias is not None
    merged = nn.Linear(layer.base.in_features, layer.base.out_features, bias=has_bias)
    merged = merged.to(device=layer.base.qweight.device, dtype=torch.float32)

    with torch.no_grad():
        base_weight = layer.base.dequantize_weight()
        delta_weight = layer.lora_B.weight @ layer.lora_A.weight
        merged.weight.copy_(base_weight + layer.scaling * delta_weight)
        if has_bias:
            merged.bias.copy_(layer.base.bias)

    return merged
