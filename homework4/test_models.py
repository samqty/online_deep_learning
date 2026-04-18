"""
Quick test to verify models can run forward passes
"""

import torch
from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner

# Test parameters
batch_size = 4
n_track = 10
n_waypoints = 3

# Create dummy inputs
track_left = torch.randn(batch_size, n_track, 2)
track_right = torch.randn(batch_size, n_track, 2)
image = torch.rand(batch_size, 3, 96, 128)  # values in [0, 1]

# Test MLPPlanner
print("Testing MLPPlanner...")
mlp_model = MLPPlanner(n_track=n_track, n_waypoints=n_waypoints)
mlp_output = mlp_model(track_left=track_left, track_right=track_right)
print(f"  Output shape: {mlp_output.shape}")
assert mlp_output.shape == (batch_size, n_waypoints, 2), "MLPPlanner output shape mismatch"
print("  ✓ MLPPlanner works!")

# Test TransformerPlanner
print("\nTesting TransformerPlanner...")
transformer_model = TransformerPlanner(n_track=n_track, n_waypoints=n_waypoints)
transformer_output = transformer_model(track_left=track_left, track_right=track_right)
print(f"  Output shape: {transformer_output.shape}")
assert transformer_output.shape == (batch_size, n_waypoints, 2), "TransformerPlanner output shape mismatch"
print("  ✓ TransformerPlanner works!")

# Test CNNPlanner
print("\nTesting CNNPlanner...")
cnn_model = CNNPlanner(n_waypoints=n_waypoints)
cnn_output = cnn_model(image=image)
print(f"  Output shape: {cnn_output.shape}")
assert cnn_output.shape == (batch_size, n_waypoints, 2), "CNNPlanner output shape mismatch"
print("  ✓ CNNPlanner works!")

print("\n✓ All models working correctly!")
