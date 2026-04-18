"""
Usage:
    python3 -m homework.train_planner --model mlp_planner --exp_dir logs
    python3 -m homework.train_planner --model transformer_planner --exp_dir logs
    python3 -m homework.train_planner --model cnn_planner --exp_dir logs
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .datasets import load_data
from .metrics import PlannerMetric
from .models import load_model, save_model


def collate_batch(batch):
    """
    Custom collate function to handle numpy arrays and convert them to tensors
    """
    result = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        # Convert numpy arrays to tensors
        if isinstance(values[0], np.ndarray):
            values = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in values]
        elif isinstance(values[0], (bool, np.bool_)):
            # Handle boolean arrays/values
            values = [torch.tensor(v, dtype=torch.bool) if isinstance(v, (bool, np.bool_)) else v for v in values]
        
        # Stack into batch
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    
    return result


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    dataset_path: str = "drive_data",
    num_epoch: int = 100,  # Increased from 50
    lr: float = 3e-4,      # Adjusted learning rate
    batch_size: int = 32,
    seed: int = 2024,
    transform_pipeline: str = "default",
    num_workers: int = 2,
    val_split: float = 0.1,
    weight_decay: float = 1e-4,  # Added weight decay
    patience: int = 20,          # Early stopping patience
    **kwargs,
):
    """
    Train a planner model.
    
    Args:
        exp_dir: Directory to save logs and models
        model_name: Model to train ('mlp_planner', 'transformer_planner', 'cnn_planner')
        dataset_path: Path to drive_data directory
        num_epoch: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        seed: Random seed
        transform_pipeline: Data transformation pipeline ('default', 'state_only', 'aug')
        num_workers: Number of data loading workers
        val_split: Fraction of data to use for validation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    # Load model
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Verify dataset exists
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path!r}. "
                                f"Please download from the README instructions.")

    # Load data - split into train/val
    dataset_path_train = dataset_path / "train"
    dataset_path_val = dataset_path / "val"
    
    if dataset_path_train.exists():
        train_data = load_data(
            str(dataset_path_train),
            transform_pipeline=transform_pipeline,
            return_dataloader=True,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
        )
        # Replace collate function with custom one
        train_data.collate_fn = collate_batch
    else:
        raise FileNotFoundError(f"Training data not found at {dataset_path_train!r}")

    if dataset_path_val.exists():
        val_data = load_data(
            str(dataset_path_val),
            transform_pipeline=transform_pipeline,
            return_dataloader=True,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
        )
        # Replace collate function with custom one
        val_data.collate_fn = collate_batch
    else:
        val_data = None

    # Loss function: L1 for waypoint regression and metric alignment
    loss_fn = torch.nn.L1Loss(reduction="none")

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=num_epoch, 
        steps_per_epoch=len(train_data), pct_start=0.1
    )

    global_step = 0
    best_val_l1 = float('inf')
    patience_counter = 0

    # Initialize metrics dictionary
    metrics = {
        "train_loss": [],
        "val_l1_error": [],
        "val_longitude": [],
        "val_lateral": []
    }

    # Training loop
    for epoch in range(num_epoch):
        # Clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        # Training phase
        for batch_idx, batch in enumerate(train_data):
            # Get inputs based on model type
            if model_name == "cnn_planner":
                # CNN model expects image input
                if "image" in batch:
                    x = batch["image"].to(device)
                else:
                    raise KeyError("Image not found in batch. Use 'default' transform pipeline.")
                y = batch["waypoints"].to(device)
            else:
                # MLP and Transformer models expect track boundaries
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                y = batch["waypoints"].to(device)
                y_mask = batch["waypoints_mask"].to(device)

            # Forward pass
            if model_name == "cnn_planner":
                pred = model(image=x)
            else:
                pred = model(track_left=track_left, track_right=track_right)

            # Compute loss - only on valid waypoints
            if model_name == "cnn_planner":
                loss = loss_fn(pred, y).mean()
            else:
                # Mask out invalid waypoints for loss computation
                masked_pred = pred * y_mask[:, :, None]
                masked_y = y * y_mask[:, :, None]
                masked_error = loss_fn(masked_pred, masked_y)
                if model_name == "transformer_planner":
                    lon_loss = masked_error[..., 0].sum()
                    lat_loss = masked_error[..., 1].sum()
                    loss = (lon_loss + 2.5 * lat_loss) / y_mask.sum()
                else:
                    loss = masked_error.sum() / (y_mask.sum() * 2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Log training loss
            logger.add_scalar("train/loss", loss.item(), global_step)
            metrics["train_loss"].append(loss.item())

            global_step += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Step learning rate scheduler
        scheduler.step()

        # Validation phase
        if val_data is not None:
            with torch.inference_mode():
                model.eval()
                val_metric = PlannerMetric()

                for batch in val_data:
                    # Get inputs
                    if model_name == "cnn_planner":
                        x = batch["image"].to(device)
                    else:
                        track_left = batch["track_left"].to(device)
                        track_right = batch["track_right"].to(device)
                        y_mask = batch["waypoints_mask"].to(device)

                    y = batch["waypoints"].to(device)

                    # Forward pass
                    if model_name == "cnn_planner":
                        pred = model(image=x)
                    else:
                        pred = model(track_left=track_left, track_right=track_right)

                    # Add to metrics (mask not used for CNN, but add it for consistency)
                    if model_name == "cnn_planner":
                        # Create mask of all True for CNN
                        mask = torch.ones_like(y[..., 0], dtype=torch.bool)
                        val_metric.add(pred, y, mask)
                    else:
                        val_metric.add(pred, y, y_mask)

                # Compute validation metrics
                val_results = val_metric.compute()
                current_val_l1 = val_results["l1_error"]
                
                print(
                    f"Epoch {epoch+1}, Val L1: {current_val_l1:.4f}, "
                    f"Lon: {val_results['longitudinal_error']:.4f}, "
                    f"Lat: {val_results['lateral_error']:.4f}"
                )

                logger.add_scalar("val/l1_error", val_results["l1_error"], epoch)
                logger.add_scalar("val/longitudinal_error", val_results["longitudinal_error"], epoch)
                logger.add_scalar("val/lateral_error", val_results["lateral_error"], epoch)

                metrics["val_l1_error"].append(val_results["l1_error"])
                metrics["val_longitude"].append(val_results["longitudinal_error"])
                metrics["val_lateral"].append(val_results["lateral_error"])

                # Early stopping check
                if current_val_l1 < best_val_l1:
                    best_val_l1 = current_val_l1
                    patience_counter = 0
                    # Save best model
                    best_model_path = log_dir / f"{model_name}_best.th"
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Print epoch summary
        avg_train_loss = np.mean(metrics["train_loss"])
        print(f"Epoch {epoch+1}/{num_epoch} - Avg Train Loss: {avg_train_loss:.4f}")
        logger.add_scalar("epoch/train_loss", avg_train_loss, epoch)

    # Save model
    save_path = save_model(model)
    print(f"Model saved to {save_path}")

    logger.close()
    print(f"Training complete. Logs saved to {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a planner model")
    parser.add_argument("--model", type=str, default="mlp_planner",
                        help="Model to train: mlp_planner, transformer_planner, cnn_planner")
    parser.add_argument("--exp_dir", type=str, default="logs",
                        help="Directory to save logs and models")
    parser.add_argument("--dataset_path", type=str, default="drive_data",
                        help="Path to dataset")
    parser.add_argument("--num_epoch", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed")
    parser.add_argument("--transform", type=str, default="default",
                        help="Data transformation pipeline (default, state_only, aug)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loading workers")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for regularization")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")

    args = parser.parse_args()

    train(
        exp_dir=args.exp_dir,
        model_name=args.model,
        dataset_path=args.dataset_path,
        num_epoch=args.num_epoch,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        transform_pipeline=args.transform,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
