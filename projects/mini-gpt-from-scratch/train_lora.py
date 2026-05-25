from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from lora_checkpoint import save_lora_adapter
from lora_minigpt import count_parameters, replace_qkv_with_lora
from model import MiniGPT
from train import TRAIN_CORPUS, VAL_CORPUS, get_batch


def load_torch_file(path: str | Path, device: str) -> Mapping[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_lora_model_from_checkpoint(
    checkpoint_path: str | Path,
    r: int = 8,
    alpha: float = 16,
    device: str = "cpu",
) -> tuple[MiniGPT, Mapping[str, object]]:
    checkpoint = load_torch_file(checkpoint_path, device="cpu")
    config = dict(checkpoint["config"])

    model = MiniGPT(**config)
    model.load_state_dict(checkpoint["model"])
    model = replace_qkv_with_lora(model, r=r, alpha=alpha)
    model.to(resolve_device(device))
    return model, checkpoint


def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def make_lora_optimizer(
    model: nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(get_trainable_parameters(model), lr=learning_rate)


def encode_corpus(corpus: str, stoi: Mapping[str, int]) -> torch.Tensor:
    ids = [stoi[ch] for ch in corpus if ch in stoi]
    if not ids:
        raise ValueError("corpus has no characters from checkpoint vocabulary")
    return torch.tensor(ids, dtype=torch.long)


def check_data_length(data: torch.Tensor, block_size: int, name: str) -> None:
    if len(data) <= block_size + 1:
        raise ValueError(f"{name} data must be longer than block_size + 1")


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    model.train()
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def estimate_lora_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> tuple[float, float]:
    model.eval()
    losses = []
    for data in (train_data, val_data):
        split_losses = torch.empty(eval_iters)
        for i in range(eval_iters):
            input_ids, targets = get_batch(data, block_size, batch_size, device)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            split_losses[i] = loss.item()
        losses.append(split_losses.mean().item())
    model.train()
    return losses[0], losses[1]


def train_lora_adapter(
    base_checkpoint: str | Path,
    adapter_output: str | Path,
    train_data: torch.Tensor | None = None,
    val_data: torch.Tensor | None = None,
    max_steps: int = 200,
    eval_interval: int = 50,
    patience: int = 3,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    eval_iters: int = 20,
    r: int = 8,
    alpha: float = 16,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, float | int]:
    torch.manual_seed(seed)
    device = resolve_device(device)
    model, checkpoint = build_lora_model_from_checkpoint(
        base_checkpoint,
        r=r,
        alpha=alpha,
        device=device,
    )

    config = dict(checkpoint["config"])
    block_size = int(config["block_size"])
    if train_data is None:
        train_data = encode_corpus(TRAIN_CORPUS, checkpoint["stoi"])
    if val_data is None:
        val_data = encode_corpus(VAL_CORPUS, checkpoint["stoi"])
    check_data_length(train_data, block_size, "train")
    check_data_length(val_data, block_size, "validation")

    total_params, trainable_params = count_parameters(model)
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    optimizer = make_lora_optimizer(model, learning_rate=learning_rate)

    print(f"base checkpoint: {base_checkpoint}")
    print(f"adapter output:  {adapter_output}")
    print(f"total parameters: {total_params}")
    print(f"trainable parameters: {trainable_params}")
    print("trainable names:")
    for name in trainable_names:
        print(f"  {name}")

    best_val_loss = float("inf")
    bad_eval_count = 0
    last_train_loss = float("nan")
    last_val_loss = float("nan")

    for step in range(max_steps):
        input_ids, targets = get_batch(train_data, block_size, batch_size, device)
        last_train_loss = train_one_step(model, optimizer, input_ids, targets)

        if step % eval_interval == 0 or step == max_steps - 1:
            train_loss, val_loss = estimate_lora_loss(
                model,
                train_data,
                val_data,
                block_size,
                batch_size,
                eval_iters,
                device,
            )
            last_train_loss = train_loss
            last_val_loss = val_loss
            print(
                f"step={step:04d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_eval_count = 0
                save_lora_adapter(model, adapter_output)
                print(f"saved best LoRA adapter: val_loss={best_val_loss:.4f}")
            else:
                bad_eval_count += 1
                print(f"validation loss did not improve: {bad_eval_count}/{patience}")
                if bad_eval_count >= patience:
                    print("early stopping: validation loss did not improve")
                    break

    return {
        "best_val_loss": best_val_loss,
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=script_dir / "runs" / "mini_gpt_best.pt",
    )
    parser.add_argument(
        "--adapter-output",
        type=Path,
        default=script_dir / "runs" / "mini_gpt_lora_adapter.pt",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_lora_adapter(
        base_checkpoint=args.base_checkpoint,
        adapter_output=args.adapter_output,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        patience=args.patience,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        r=args.r,
        alpha=args.alpha,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
