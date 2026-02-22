"""
Direct CNN model for event-based pose estimation (SPADES paper baseline).
Uses ResNet-18 backbone for frame-by-frame pose prediction without temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Tuple, Optional


class DirectPoseCNN(nn.Module):
    """
    Direct frame-by-frame CNN for pose estimation (matches SPADES paper's Direct approach).
    
    Architecture:
        1. CNN Backbone (ResNet-18): Extract spatial features from event frame
        2. Dual Regression Heads: Predict translation and rotation directly
    
    Args:
        num_input_channels: Number of event frame channels (3 for exponential decay, 5 for voxel)
        dropout: Dropout probability (default: 0.2)
        pretrained_backbone: Use ImageNet pretrained weights (default: True)
    """
    
    def __init__(
        self,
        num_input_channels: int = 3,
        dropout: float = 0.2,
        pretrained_backbone: bool = True
    ):
        super(DirectPoseCNN, self).__init__()
        
        self.num_input_channels = num_input_channels
        
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
            For non-RGB channels, initialize conv1 from pretrained weights
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
            # For 3 channels, use pretrained RGB weights directly (BEST for 3-channel representation!)
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
        Dual regression heads (directly from CNN features, no LSTM)
        # Translation head: predict [Tx, Ty, Tz]
        self.translation_head = nn.Sequential(
            nn.Linear(cnn_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        
        # Rotation head: predict [Qw, Qx, Qy, Qz] (quaternion)
        self.rotation_head = nn.Sequential(
            nn.Linear(cnn_output_dimtial(
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
        event_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through full model (frame-by-frame prediction).
        
        Args:
            event_frames: (batch_size, num_channels, height, width) - Single frames
                         OR (batch_size, 1, num_channels, height, width) if sequence_length=1
            
        Returns:
            translation: (batch_size, 3) - [Tx, Ty, Tz]
            rotation: (batch_size, 4) - [Qw, Qx, Qy, Qz] normalized
        """
        # Handle sequence dimension if present (batch_size, 1, C, H, W) -> (batch_size, C, H, W)
        if event_frames.dim() == 5 and event_frames.size(1) == 1:
            event_frames = event_frames.squeeze(1)
        
        # Extract CNN features
        features = self.forward_cnn(event_frames)  # (batch_size, 512)
        
        # Regression heads
        translation = self.translation_head(features)  # (batch_size, 3)
        rotation = self.rotation_head(features)        # (batch_size, 4)
        
        # Normalize quaternion
        rotation = F.normalize(rotation, p=2, dim=1)
        
        return translation, rotation
    
    def predict(self, event_frames: torch.Tensor) -> torch.Tensor:
        """
        Predict pose for inference (convenience method).
        
        Returns concatenated pose [Tx, Ty, Tz, Qw, Qx, Qy, Qz].
        """
        translation, rotation = self.forward(event_frames)
        pose = torch.cat([translation, rotation], dim=-1)
        return pose


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model forward pass with dummy data."""
    print("Testing DirectPoseCNN model...")
    
    # Create dummy input (single frames, not sequences)
    batch_size = 8
    num_channels = 3
    height, width = 720, 1280
    
    dummy_input = torch.randn(batch_size, num_channels, height, width)
    
    # Create model
    model = DirectPoseCNN(num_input_channels=3)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        translation, rotation = model(dummy_input)
    
    # Check outputs
    print(f"Input shape: {dummy_input.shape}")
    print(f"Translation shape: {translation.shape}")  # (8, 3)
    print(f"Rotation shape: {rotation.shape}")        # (8, 4)
    print(f"Quaternion norms: {torch.norm(rotation, dim=-1)}")  # Should be ~1.0
    
    # Check parameter count
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")
    
    print("Model test passed!")


if __name__ == "__main__":
    test_model()
