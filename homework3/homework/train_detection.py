import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data as load_road_data
from homework.metrics import DetectionMetric


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def train_detection(
    exp_dir: str = "logs",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 2024,
    num_workers: int = 4,
    seg_weight: float = 1.0,
    depth_weight: float = 1.0,
    in_channels: int = 3,
    num_classes: int = 3,
):
    device = get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    root_dir = Path(__file__).resolve().parent.parent
    train_path = root_dir / "drive_data" / "train"
    val_path = root_dir / "drive_data" / "val"

    if not train_path.exists():
        raise FileNotFoundError(f"training data not found at {train_path!r}")
    if not val_path.exists():
        raise FileNotFoundError(f"validation data not found at {val_path!r}")

    log_dir = Path(exp_dir) / f"detector_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = tb.SummaryWriter(log_dir)

    model = Detector(in_channels=in_channels, num_classes=num_classes).to(device)

    train_loader = load_road_data(train_path, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = load_road_data(val_path, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    seg_criterion = torch.nn.CrossEntropyLoss()
    depth_criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metric = DetectionMetric(num_classes=num_classes)
    global_step = 0

    for epoch in range(1, num_epoch + 1):
        model.train()
        metric.reset()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_depth_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            track = batch["track"].to(device)
            depth = batch["depth"].to(device)

            optimizer.zero_grad()
            logits, depth_pred = model(images)

            seg_loss = seg_criterion(logits, track.long())
            depth_loss = depth_criterion(depth_pred, depth)
            loss = seg_weight * seg_loss + depth_weight * depth_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_seg_loss += seg_loss.item() * images.size(0)
            total_depth_loss += depth_loss.item() * images.size(0)

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                metric.add(pred, track, depth_pred, depth)

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/seg_loss", seg_loss.item(), global_step)
            writer.add_scalar("train/depth_loss", depth_loss.item(), global_step)
            global_step += 1

        train_metrics = metric.compute()

        n_train = len(train_loader.dataset)
        epoch_loss = total_loss / n_train
        epoch_seg_loss = total_seg_loss / n_train
        epoch_depth_loss = total_depth_loss / n_train

        model.eval()
        metric.reset()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_depth_loss = 0.0

        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device)
                track = batch["track"].to(device)
                depth = batch["depth"].to(device)

                logits, depth_pred = model(images)

                seg_loss = seg_criterion(logits, track.long())
                depth_loss = depth_criterion(depth_pred, depth)
                loss = seg_weight * seg_loss + depth_weight * depth_loss

                val_loss += loss.item() * images.size(0)
                val_seg_loss += seg_loss.item() * images.size(0)
                val_depth_loss += depth_loss.item() * images.size(0)

                pred = logits.argmax(dim=1)
                metric.add(pred, track, depth_pred, depth)

        val_metrics = metric.compute()

        n_val = len(val_loader.dataset)

        writer.add_scalar("epoch/train_loss", epoch_loss, epoch)
        writer.add_scalar("epoch/train_seg_loss", epoch_seg_loss, epoch)
        writer.add_scalar("epoch/train_depth_loss", epoch_depth_loss, epoch)
        writer.add_scalar("epoch/train_iou", train_metrics["iou"], epoch)
        writer.add_scalar("epoch/train_accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("epoch/val_loss", val_loss / n_val, epoch)
        writer.add_scalar("epoch/val_seg_loss", val_seg_loss / n_val, epoch)
        writer.add_scalar("epoch/val_depth_loss", val_depth_loss / n_val, epoch)
        writer.add_scalar("epoch/val_iou", val_metrics["iou"], epoch)
        writer.add_scalar("epoch/val_accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("epoch/val_abs_depth_error", val_metrics["abs_depth_error"], epoch)
        writer.add_scalar("epoch/val_tp_depth_error", val_metrics["tp_depth_error"], epoch)

        print(
            f"Epoch {epoch}/{num_epoch}: "
            f"train_loss={epoch_loss:.4f}, val_loss={val_loss/n_val:.4f}, "
            f"val_iou={val_metrics['iou']:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
            f"val_depth={val_metrics['abs_depth_error']:.4f}"
        )

        if epoch % 5 == 0 or epoch == num_epoch:
            torch.save(model.state_dict(), log_dir / f"detector_epoch_{epoch}.pth")

    model_cpu = model.cpu()
    output_path = save_model(model_cpu)
    torch.save(model_cpu.state_dict(), log_dir / "detector.th")
    print(f"✓ Detector saved to {output_path}")
    print(f"✓ Checkpoint saved to {log_dir / 'detector.th'}")

    writer.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Detection training script")
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--in_channels", type=int, default=3)
    args = parser.parse_args()

    train_detection(
        exp_dir=args.exp_dir,
        num_epoch=args.num_epoch,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        seg_weight=args.seg_weight,
        depth_weight=args.depth_weight,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
