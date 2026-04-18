"""
Quick validation script to test model loading and forward passes
"""

import sys
from pathlib import Path

import torch

# Add homework directory to path
sys.path.insert(0, str(Path(__file__).parent))

from homework.models import MODEL_FACTORY, load_model, save_model


def test_model_sizes():
    """Verify all models are under 20MB limit"""
    print("Testing model sizes...")
    for model_name in MODEL_FACTORY.keys():
        model = load_model(model_name)
        size_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
        print(f"  {model_name}: {size_mb:.2f} MB")
        assert size_mb < 20, f"{model_name} exceeds 20MB limit"
    print("✓ All models under size limit\n")


def test_mlp_planner():
    """Test MLPPlanner forward pass"""
    print("Testing MLPPlanner...")
    model = load_model("mlp_planner")
    model.eval()
    
    batch_size, n_track, n_waypoints = 4, 10, 3
    track_left = torch.randn(batch_size, n_track, 2)
    track_right = torch.randn(batch_size, n_track, 2)
    
    with torch.no_grad():
        output = model(track_left=track_left, track_right=track_right)
    
    assert output.shape == (batch_size, n_waypoints, 2), f"Wrong shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    print(f"  Output shape: {output.shape} ✓\n")


def test_transformer_planner():
    """Test TransformerPlanner forward pass"""
    print("Testing TransformerPlanner...")
    model = load_model("transformer_planner")
    model.eval()
    
    batch_size, n_track, n_waypoints = 4, 10, 3
    track_left = torch.randn(batch_size, n_track, 2)
    track_right = torch.randn(batch_size, n_track, 2)
    
    with torch.no_grad():
        output = model(track_left=track_left, track_right=track_right)
    
    assert output.shape == (batch_size, n_waypoints, 2), f"Wrong shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    print(f"  Output shape: {output.shape} ✓\n")


def test_cnn_planner():
    """Test CNNPlanner forward pass"""
    print("Testing CNNPlanner...")
    model = load_model("cnn_planner")
    model.eval()
    
    batch_size, n_waypoints = 4, 3
    # Image values in [0, 1]
    image = torch.rand(batch_size, 3, 96, 128)
    
    with torch.no_grad():
        output = model(image=image)
    
    assert output.shape == (batch_size, n_waypoints, 2), f"Wrong shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    print(f"  Output shape: {output.shape} ✓\n")


def test_model_saving():
    """Test model saving and loading"""
    print("Testing model save/load...")
    
    for model_name in MODEL_FACTORY.keys():
        model = load_model(model_name)
        
        # Save model
        path = save_model(model)
        print(f"  Saved {model_name} to {path}")
        
        # Load model with weights
        loaded_model = load_model(model_name, with_weights=True)
        
        # Verify weights are same
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2), f"Weights mismatch for {model_name}"
    
    print("✓ Model save/load working\n")


def test_loss_function():
    """Test loss computation"""
    print("Testing loss function...")
    
    batch_size, n_waypoints = 4, 3
    pred = torch.randn(batch_size, n_waypoints, 2)
    target = torch.randn(batch_size, n_waypoints, 2)
    mask = torch.ones(batch_size, n_waypoints, dtype=torch.bool)
    
    # MSE loss
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(pred, target)
    
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "NaN in loss"
    print(f"  Loss: {loss.item():.4f} ✓")
    
    # Masked loss
    masked_loss = loss_fn(pred * mask[..., None], target * mask[..., None])
    assert masked_loss.item() >= 0, "Masked loss should be non-negative"
    print(f"  Masked loss: {masked_loss.item():.4f} ✓\n")


def test_metrics():
    """Test metrics computation"""
    print("Testing metrics...")
    from homework.metrics import PlannerMetric
    
    batch_size, n_waypoints = 4, 3
    pred = torch.randn(batch_size, n_waypoints, 2)
    target = torch.randn(batch_size, n_waypoints, 2)
    mask = torch.ones(batch_size, n_waypoints, dtype=torch.bool)
    
    metric = PlannerMetric()
    metric.add(pred, target, mask)
    results = metric.compute()
    
    assert "l1_error" in results, "Missing l1_error"
    assert "longitudinal_error" in results, "Missing longitudinal_error"
    assert "lateral_error" in results, "Missing lateral_error"
    assert "num_samples" in results, "Missing num_samples"
    
    print(f"  L1 error: {results['l1_error']:.4f}")
    print(f"  Longitudinal: {results['longitudinal_error']:.4f}")
    print(f"  Lateral: {results['lateral_error']:.4f}")
    print(f"  Samples: {results['num_samples']} ✓\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Running validation tests for Homework 4")
    print("=" * 60 + "\n")
    
    try:
        test_model_sizes()
        test_mlp_planner()
        test_transformer_planner()
        test_cnn_planner()
        test_model_saving()
        test_loss_function()
        test_metrics()
        
        print("=" * 60)
        print("✓ All validation tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
