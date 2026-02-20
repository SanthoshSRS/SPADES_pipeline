"""
CNN+LSTM model for event-based pose estimation.
Uses ResNet-18 backbone for spatial features and LSTM for temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Tuple, Optional


class VoxelCNNLSTM(nn.Module):
    """
    End-to-end CNN+LSTM model for pose estimation from event sequences.
    
    Architecture:
        1. CNN Backbone (ResNet-18): Extract spatial features from each voxel grid
        2. LSTM: Model temporal dependencies across sequence
        3. Dual Regression Heads: Predict translation and rotation separately
    
    Args:
        num_input_channels: Number of voxel grid channels (5 for signed, 10 for standard)
        lstm_hidden_size: LSTM hidden dimension (default: 256)
        lstm_num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.2)
        pretrained_backbone: Use ImageNet pretrained weights (default: True)
    """
    
    def __init__(
        self,
        num_input_channels: int = 5,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
        pretrained_backbone: bool = True
    ):
        super(VoxelCNNLSTM, self).__init__()
        
        self.num_input_channels = num_input_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # CNN Backbone: ResNet-18
        if pretrained_backbone:
            weights = ResNet18_Weights.IMAGENET1K_V1
            resnet = resnet18(weights=weights)
        else:
            resnet = resnet18(weights=None)
        
        # Modify first conv layer for event channels
        if num_input_channels != 3:
            # Initialize 5-channel conv1 from 3-channel RGB weights
            original_conv1 = resnet.conv1
            
            # Create new conv1 with correct input channels
            self.conv1 = nn.Conv2d(
                num_input_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Initialize: replicate RGB weights across channels
            with torch.no_grad():
                if pretrained_backbone:
                    # Average RGB weights and replicate
                    rgb_weights = original_conv1.weight.data  # (64, 3, 7, 7)
                    avg_weights = rgb_weights.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                    self.conv1.weight.data = avg_weights.repeat(1, num_input_channels, 1, 1)
                else:
                    # Random initialization
                    nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        else:
            self.conv1 = resnet.conv1
        
        # Keep rest of ResNet backbone
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # CNN output: 512-dimensional feature vector per frame
        cnn_output_dim = 512
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )
        
        # Dual regression heads
        # Translation head: predict [Tx, Ty, Tz]
        self.translation_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        
        # Rotation head: predict [Qw, Qx, Qy, Qz] (quaternion)
        self.rotation_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 4)
        )
        
    def forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN backbone.
        
        Args:
            x: Input voxel grid (batch_size, num_channels, height, width)
            
        Returns:
            features: Spatial feature vector (batch_size, 512)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(
        self,
        voxel_sequence: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through full model.
        
        Args:
            voxel_sequence: (batch_size, sequence_length, num_channels, height, width)
            hidden_state: Optional LSTM hidden state for sequential inference
            
        Returns:
            translation: (batch_size, sequence_length, 3) - [Tx, Ty, Tz]
            rotation: (batch_size, sequence_length, 4) - [Qw, Qx, Qy, Qz] normalized
        """
        batch_size, seq_len, num_channels, height, width = voxel_sequence.shape
        
        # Reshape for CNN: (batch_size * seq_len, num_channels, height, width)
        voxel_flat = voxel_sequence.view(batch_size * seq_len, num_channels, height, width)
        
        # Extract CNN features for all frames
        features = self.forward_cnn(voxel_flat)  # (batch_size * seq_len, 512)
        
        # Reshape for LSTM: (batch_size, seq_len, 512)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM temporal modeling
        if hidden_state is not None:
            lstm_out, hidden_state = self.lstm(features, hidden_state)
        else:
            lstm_out, _ = self.lstm(features)  # (batch_size, seq_len, lstm_hidden_size)
        
        # Regression heads
        # Reshape for heads: (batch_size * seq_len, lstm_hidden_size)
        lstm_flat = lstm_out.contiguous().view(batch_size * seq_len, -1)
        
        translation = self.translation_head(lstm_flat)  # (batch_size * seq_len, 3)
        rotation = self.rotation_head(lstm_flat)        # (batch_size * seq_len, 4)
        
        # Normalize quaternion
        rotation = F.normalize(rotation, p=2, dim=1)
        
        # Reshape back to sequences
        translation = translation.view(batch_size, seq_len, 3)
        rotation = rotation.view(batch_size, seq_len, 4)
        
        return translation, rotation
    
    def predict(self, voxel_sequence: torch.Tensor) -> torch.Tensor:
        """
        Predict pose for inference (convenience method).
        
        Args:
            voxel_sequence: (batch_size, sequence_length, num_channels, height, width)
            
        Returns:
            pose: (batch_size, sequence_length, 7) - [Tx, Ty, Tz, Qw, Qx, Qy, Qz]
        """
        translation, rotation = self.forward(voxel_sequence)
        
        # Concatenate: [Tx, Ty, Tz, Qw, Qx, Qy, Qz]
        pose = torch.cat([translation, rotation], dim=-1)
        
        return pose


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model forward pass with dummy data."""
    print("Testing VoxelCNNLSTM model...")
    
    # Create dummy input
    batch_size = 4
    seq_len = 10
    num_channels = 5
    height, width = 720, 1280
    
    dummy_input = torch.randn(batch_size, seq_len, num_channels, height, width)
    
    # Create model
    model = VoxelCNNLSTM(num_input_channels=5)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        translation, rotation = model(dummy_input)
    
    # Check outputs
    print(f"Input shape: {dummy_input.shape}")
    print(f"Translation shape: {translation.shape}")  # (4, 10, 3)
    print(f"Rotation shape: {rotation.shape}")        # (4, 10, 4)
    print(f"Quaternion norms: {torch.norm(rotation, dim=-1)[0]}")  # Should be ~1.0
    
    # Check parameter count
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")
    
    print("Model test passed!")


if __name__ == "__main__":
    test_model()
