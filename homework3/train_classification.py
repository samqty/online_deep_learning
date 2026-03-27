import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.models import ClassificationLoss, load_model, save_model
from homework.datasets.classification_dataset import load_data
from homework.metrics import compute_accuracy


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def train_classification(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    num_workers: int = 4,
    **kwargs,
):
    device = get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)

    train_loader = load_data(
        "classification_data/train",
        transform_pipeline="aug",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = load_data(
        "classification_data/val",
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    criterion = ClassificationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(1, num_epoch + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            acc = compute_accuracy(logits, y)
            running_loss += loss.item() * x.size(0)
            running_acc += acc.item() * x.size(0)

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/acc", acc.item(), global_step)
            global_step += 1

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_acc / len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                acc = compute_accuracy(logits, y)

                val_loss += loss.item() * x.size(0)
                val_acc += acc.item() * x.size(0)

        val_loss /= len(val_dataset)
        val_acc /= len(val_dataset)

        writer.add_scalar("epoch/train_loss", epoch_loss, epoch)
        writer.add_scalar("epoch/train_acc", epoch_acc, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/val_acc", val_acc, epoch)

        print(
            f"Epoch {epoch}/{num_epoch}: "
            f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if epoch % 10 == 0 or epoch == num_epoch:
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}_epoch_{epoch}.pth")

    writer.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Classification training script")
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="linear")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    args = parser.parse_args()

    train_classification(
        exp_dir=args.exp_dir,
        model_name=args.model_name,
        num_epoch=args.num_epoch,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    main()
