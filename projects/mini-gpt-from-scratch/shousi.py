import torch
from torch import nn
def freeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

def replace_qkv_with_lora(model, r=8, alpha=16):
    freeze_all_parameters(model)
    for block in model.parameters():
        block.attn.qkv = lora_linear(block.attn.qkv,r = r,alpha = alpha)
    return model

