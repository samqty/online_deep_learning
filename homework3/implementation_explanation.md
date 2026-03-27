# Implementation Explanation for models.py

This document provides a detailed explanation of the implementation of the `Classifier` and `Detector` models in `models.py`, based on the requirements outlined in the README.md for Homework 3.

## Overview

The homework involves implementing two convolutional neural network models using PyTorch:
1. **Classifier**: A convolutional network for image classification on the SuperTuxKart dataset (6 classes, 64x64 images).
2. **Detector**: A U-Net style network for simultaneous semantic segmentation and depth regression on the SuperTuxKart Drive dataset (3 classes for segmentation, depth prediction on 96x128 images).

Both models include input normalization using pre-computed mean and standard deviation values.

## Classifier Model

### Architecture

The `Classifier` is designed as a simple convolutional neural network suitable for image classification tasks. The architecture consists of:

1. **Convolutional Blocks**: Three convolutional layers with increasing channel depth (32, 64, 128 channels) followed by batch normalization and ReLU activation.
2. **Pooling**: Max pooling (2x2) after each convolutional block to reduce spatial dimensions.
3. **Global Average Pooling**: Adaptive average pooling to reduce feature maps to 1x1 spatial size.
4. **Fully Connected Layer**: A linear layer that maps the flattened features to class logits.

### Detailed Layer Configuration

- **Conv1**: `nn.Conv2d(3, 32, kernel_size=3, padding=1)` - Maintains spatial size with padding
- **BatchNorm1**: `nn.BatchNorm2d(32)` - Normalizes activations for stable training
- **Pool**: `nn.MaxPool2d(2, 2)` - Reduces spatial dimensions by half
- **Conv2**: `nn.Conv2d(32, 64, kernel_size=3, padding=1)`
- **BatchNorm2**: `nn.BatchNorm2d(64)`
- **Conv3**: `nn.Conv2d(64, 128, kernel_size=3, padding=1)`
- **BatchNorm3**: `nn.BatchNorm2d(128)`
- **AdaptiveAvgPool2d(1)**: Global average pooling
- **FC**: `nn.Linear(128, num_classes)` - Classification head

### Forward Pass

1. Input normalization: `(x - mean) / std`
2. Three conv-pool blocks: Conv → BatchNorm → ReLU → MaxPool
3. Global average pooling to get 128 features per sample
4. Linear classification to get logits

### Design Choices

- **Depth**: Three convolutional layers provide sufficient capacity for the 64x64 classification task without excessive parameters.
- **Batch Normalization**: Improves training stability and convergence speed.
- **Global Average Pooling**: Reduces overfitting compared to flattening and provides spatial invariance.
- **No Dropout**: Not included as the model is relatively simple and global pooling helps prevent overfitting.
- **Activation**: ReLU for non-linearity, standard for CNNs.

### Expected Performance

This architecture should achieve >80% validation accuracy as required, with potential for >90% with proper training and data augmentation.

## Detector Model

### Architecture

The `Detector` implements a U-Net style encoder-decoder architecture for simultaneous segmentation and depth prediction. It features:

1. **Encoder (Downsampling)**: Two strided convolutional blocks that reduce spatial resolution while increasing channels.
2. **Decoder (Upsampling)**: Two transposed convolutional blocks that restore spatial resolution.
3. **Skip Connections**: Concatenation of encoder features with decoder features to preserve spatial details.
4. **Dual Heads**: Separate convolutional heads for segmentation logits and depth prediction.

### Detailed Layer Configuration

#### Encoder:
- **Down1**: `nn.Conv2d(3, 16, 3, stride=2, padding=1)` → `nn.BatchNorm2d(16)` → `nn.ReLU()`
- **Down2**: `nn.Conv2d(16, 32, 3, stride=2, padding=1)` → `nn.BatchNorm2d(32)` → `nn.ReLU()`

#### Decoder:
- **Up1**: `nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)` → `nn.BatchNorm2d(16)` → `nn.ReLU()`
- **Up1_Conv**: `nn.Conv2d(32, 16, 3, padding=1)` → `nn.BatchNorm2d(16)` → `nn.ReLU()` (after concatenation)
- **Up2**: `nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)` → `nn.BatchNorm2d(16)` → `nn.ReLU()`

#### Heads:
- **Segmentation Head**: `nn.Conv2d(16, 3, 1)` - 1x1 conv for 3-class segmentation
- **Depth Head**: `nn.Conv2d(16, 1, 1)` - 1x1 conv for depth prediction

### Forward Pass

1. Input normalization: `(x - mean) / std`
2. Encoder: Two downsampling blocks (spatial reduction by 4x total)
3. Decoder: Two upsampling blocks with skip connection from first encoder block
4. Heads: Generate segmentation logits and raw depth predictions

### Skip Connections

A skip connection concatenates the output of `Down1` (16 channels) with the upsampled `Up1` output (16 channels), creating a 32-channel feature map. A subsequent 3x3 convolution reduces this back to 16 channels, allowing the network to leverage both high-level and low-level features for better boundary detection.

### Depth Post-processing

In the `predict` method, raw depth predictions are passed through a sigmoid activation to constrain outputs to the [0, 1] range, matching the normalized depth ground truth.

### Design Choices

- **U-Net Style**: Encoder-decoder with skip connections is standard for segmentation tasks and helps preserve spatial details.
- **Strided Convolutions**: Used instead of pooling for downsampling to maintain more information.
- **Transposed Convolutions**: For upsampling, with appropriate kernel size (4) and stride (2) to match spatial dimensions.
- **Channel Progression**: 3→16→32→16→16 provides sufficient capacity without excessive parameters.
- **Dual Tasks**: Single backbone with separate heads allows joint learning of segmentation and depth.
- **Batch Normalization**: Applied throughout for stable training.
- **No Additional Regularization**: Skip connections and batch norm provide sufficient regularization.

### Expected Performance

This architecture should achieve:
- Segmentation IoU > 0.75
- Depth MAE < 0.05
- Lane boundary depth MAE < 0.05

The skip connections are crucial for capturing fine boundary details in segmentation, while the shared encoder allows depth prediction to benefit from learned features.

## Training Considerations

### Loss Functions
- **Classification**: Cross-entropy loss on logits
- **Segmentation**: Cross-entropy loss on pixel-wise logits  
- **Depth**: Mean absolute error or mean squared error on depth predictions
- **Combined**: Weighted sum of segmentation and depth losses

### Data Augmentation
For classification, consider random horizontal flips and other transformations in the dataset transforms.

### Model Size
Both models are designed to stay under the 20MB limit when saved, with the Classifier being much smaller than the Detector due to fewer parameters.

## Validation

The `debug_model` function can be used to verify tensor shapes and basic functionality. Run `python models.py` to test the Classifier implementation.