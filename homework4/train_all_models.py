#!/usr/bin/env python3
"""
Train all three planner models with optimized hyperparameters
"""

import os
import subprocess
import sys
from pathlib import Path

def run_training(model_name, epochs, lr, batch_size=32, use_aug=False):
    """Run training for a specific model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")

    transform = "aug" if use_aug else "default"
    if model_name in ["mlp_planner", "transformer_planner"]:
        transform = "aug" if use_aug else "state_only"

    cmd = [
        sys.executable, "-m", "homework.train_planner",
        "--model", model_name,
        "--num_epoch", str(epochs),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--transform", transform,
        "--weight_decay", "1e-4",
        "--patience", "15"
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0

def main():
    """Train all three models with optimized settings"""

    print("🚀 Starting comprehensive model training")
    print("This will train all three models with optimized hyperparameters")

    # Model configurations
    configs = [
        {
            "name": "mlp_planner",
            "epochs": 100,
            "lr": 3e-4,
            "batch_size": 64,
            "use_aug": True
        },
        {
            "name": "transformer_planner",
            "epochs": 80,
            "lr": 5e-4,
            "batch_size": 32,
            "use_aug": True
        },
        {
            "name": "cnn_planner",
            "epochs": 120,
            "lr": 3e-4,
            "batch_size": 32,
            "use_aug": False  # CNN uses image augmentation built-in
        }
    ]

    results = {}

    for config in configs:
        success = run_training(**config)
        results[config["name"]] = success

        if not success:
            print(f"❌ {config['name']} training failed!")
        else:
            print(f"✅ {config['name']} training completed!")

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")

    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model:20} {status}")

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Check TensorBoard: tensorboard --logdir logs")
    print("2. Run grader: python3 -m grader homework -vv")
    print("3. Submit best models: python3 bundle.py homework $ID")
    print("\nExpected performance:")
    print("- MLPPlanner: Lon < 0.2, Lat < 0.6")
    print("- TransformerPlanner: Lon < 0.2, Lat < 0.6")
    print("- CNNPlanner: Lon < 0.30, Lat < 0.45")

if __name__ == "__main__":
    main()