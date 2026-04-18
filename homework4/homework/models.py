from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 256,  # Increased from 128
        dropout: float = 0.1,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): hidden dimension for MLP layers
            dropout (float): dropout probability
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Input: (B, n_track, 2) for left and right, concatenated to (B, n_track * 4)
        input_dim = n_track * 2 * 2  # left + right, each with 2 coords
        output_dim = n_waypoints * 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Added batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),  # Added dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Better weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]

        # Concatenate left and right boundaries and flatten
        # (b, n_track, 2) -> (b, n_track * 2)
        x = torch.cat([track_left, track_right], dim=-1)
        x = x.view(batch_size, -1)

        # Pass through MLP
        waypoints = self.mlp(x)

        # Reshape to (b, n_waypoints, 2)
        waypoints = waypoints.view(batch_size, self.n_waypoints, 2)

        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,  # Increased from 64
        nhead: int = 8,      # Increased from 4
        num_decoder_layers: int = 3,  # Increased from 2
        dim_feedforward: int = 512,   # Increased from 256
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Embedding for waypoint queries (latent array)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Project input track points (centerline and width) to d_model dimension
        self.input_proj = nn.Linear(4, d_model)

        # Positional encoding for track points along the track
        self.track_pos_embed = nn.Embedding(n_track, d_model)
        self.query_pos_embed = nn.Embedding(n_waypoints, d_model)

        # Transformer encoder for track context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        # Transformer decoder layer with cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Output MLP to predict 2D waypoints
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        # Better initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]

        # Compute centerline and width features for each track point
        track_center = (track_left + track_right) * 0.5
        track_width = track_right - track_left
        track_points = torch.cat([track_center, track_width], dim=-1)

        # Project track points to d_model dimension: (b, n_track, d_model)
        memory = self.input_proj(track_points)

        # Add positional embeddings along the track sequence
        track_indices = torch.arange(self.n_track, device=track_left.device)
        track_pos = self.track_pos_embed(track_indices).unsqueeze(0).expand(batch_size, -1, -1)
        memory = memory + track_pos
        # Encode track context
        memory = self.transformer_encoder(memory)
        # Get query embeddings for waypoints: (n_waypoints, d_model)
        # Expand batch dimension: (b, n_waypoints, d_model)
        query_indices = torch.arange(self.n_waypoints, device=track_left.device)
        tgt = self.query_embed(query_indices).unsqueeze(0).expand(batch_size, -1, -1)
        tgt = tgt + self.query_pos_embed(query_indices).unsqueeze(0).expand(batch_size, -1, -1)

        # Apply transformer decoder
        # tgt: (b, n_waypoints, d_model)
        # memory: (b, n_track * 2, d_model)
        output = self.transformer_decoder(tgt, memory)

        # Apply layer normalization
        output = self.norm(output)

        # Project to 2D waypoints: (b, n_waypoints, 2)
        waypoints = self.output_mlp(output)

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        n_track: int = 10,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints
        self.n_track = n_track

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Track encoder for boundary information
        self.track_encoder = nn.Sequential(
            nn.Linear(n_track * 2 * 2, 256),  # track_left + track_right flattened
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Improved CNN backbone with residual connections
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Improved fully connected layers - now takes image features + track features
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_waypoints * 2),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BottleneckBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        image: torch.Tensor,
        track_left: torch.Tensor | None = None,
        track_right: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]
            track_left (torch.Tensor, optional): shape (b, n_track, 2)
            track_right (torch.Tensor, optional): shape (b, n_track, 2)

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        batch_size = image.shape[0]

        # Encode track boundaries if provided, otherwise use a learned zero prior
        if track_left is not None and track_right is not None:
            track_combined = torch.cat([track_left, track_right], dim=-1)  # (b, n_track, 4)
            track_flat = track_combined.view(batch_size, -1)  # (b, n_track * 4)
            track_features = self.track_encoder(track_flat)  # (b, 128)
        else:
            track_features = torch.zeros(batch_size, 128, device=image.device, dtype=image.dtype)

        # Process image
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # CNN backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (b, 512)

        # Concatenate image and track features
        combined_features = torch.cat([x, track_features], dim=1)  # (b, 512 + 128)

        # Fully connected layers
        x = self.fc(combined_features)

        # Reshape to (B, n_waypoints, 2)
        waypoints = x.view(x.size(0), self.n_waypoints, 2)

        return waypoints


class BottleneckBlock(nn.Module):
    """Bottleneck residual block for deeper networks"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
