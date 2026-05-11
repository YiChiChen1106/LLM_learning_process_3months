from __future__ import annotations

import torch
from torch import nn
class LoraLinear(nn.Module):
    def init(
            self,
            in_features: int,
            out_features: int,
            r: int = 8,
            alpha: int = 16,
            bias: bool = True,
    )->None:
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive")
        self.linear = nn.Linear(in_features,out_features,bias=bias)
        for parameters in self.linear.parameters():
            parameters.requires_grad = False
        self.lora_a = nn.Linear(in_features,r,bias=False)
        self.lora_b = nn.Linear(r,out_features,bias=False)
        self.scaling = alpha / r

        nn.init.normal_(self.lora_a.weight,mean=0.00,std=0.02)
        nn.init.zeros_(self.lora_b.weight)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        base_output = self.linear(x)
        lora_output = self.lora_b(self.lora_a(x))*self.scaling
        return base_output + lora_output




