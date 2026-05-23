from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if "lora_" in name
    }


def count_lora_parameters(lora_state: Mapping[str, torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in lora_state.values())


def save_lora_adapter(model: nn.Module, path: str | Path) -> dict[str, torch.Tensor]:
    lora_state = get_lora_state_dict(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state, path)
    return lora_state


def load_lora_adapter(model: nn.Module, path: str | Path) -> nn.Module:
    try:
        lora_state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        lora_state = torch.load(path, map_location="cpu")
    model.load_state_dict(lora_state, strict=False)
    return model


def save_merged_checkpoint(
    model: nn.Module,
    source_checkpoint: Mapping[str, object],
    path: str | Path,
) -> dict[str, object]:
    merged_checkpoint = {
        "model": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "config": source_checkpoint["config"],
        "stoi": source_checkpoint["stoi"],
        "itos": source_checkpoint["itos"],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_checkpoint, path)
    return merged_checkpoint
