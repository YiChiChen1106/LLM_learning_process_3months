import torch


def main() -> None:
    torch.manual_seed(42)

    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    linear = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

    for step in range(10):
        pred = linear(x)
        loss = torch.nn.functional.mse_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(f"步骤={step:02d} 损失={loss.item():.4f}")

    weight_grad_shape = linear.weight.grad.shape
    print(f"最终权重形状={tuple(linear.weight.shape)} 梯度形状={tuple(weight_grad_shape)}")


if __name__ == "__main__":
    main()
