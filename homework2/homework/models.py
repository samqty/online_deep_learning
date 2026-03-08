"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.CrossEntropyLoss()(logits, target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * h * w, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.linear(self.flatten(x))


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3 * h * w, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class MLPClassifierDeep(nn.Module):
    def __init__(self, h: int = 64, w: int = 64, num_classes: int = 6, 
                 hidden_dim: int = 50, num_layers: int = 5):
        super().__init__()
        num_layers = max(1, num_layers)
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        
        input_features = 3 * h * w
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_features, hidden_dim))
        
        # Subsequent transitions: Hidden-to-hidden logic
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve batch dimension 'b' while collapsing (3, H, W)
        x = self.flatten(x)
        
        # Dynamic traversal of the computational graph
        for layer in self.layers:
            x = self.relu(layer(x))
            
        return self.out(x)


class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 50,
        num_layers: int = 5
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        self.num_layers = num_layers
        self.flatten = nn.Flatten()

        # Use nn.ModuleList for dynamic layer creation [2]
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(3 * h * w, hidden_dim))
        
        # Remaining hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.out = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.flatten(x)
        h_first = None
        
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1 and h_first is not None:
                # Generalizing the residual skip: Add the first hidden layer's output 
                x = self.relu(layer(x + h_first))
            else:
                x = self.relu(layer(x))
                if i == 0:
                    h_first = x
                    
        return self.out(x)


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for name, cls in model_factory.items():
        if isinstance(model, cls):
            path = Path(__file__).resolve().parent / f"{name}.th"
            torch.save(model.state_dict(), path)
            return
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path.name} not found")
        r.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise ValueError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
