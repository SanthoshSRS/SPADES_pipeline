import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from .gradient_reversal import GradientReversalLayer


class DomainAdaptivePoseNet(nn.Module):
    """
    CNN + GRU temporal model with DANN domain adaptation.

    Architecture:
        ResNet-50 (or 18) backbone  →  per-frame spatial features
        GRU temporal layer          →  sequence memory
        translation_head            →  [Tx, Ty, Tz]
        rotation_head               →  [Qw, Qx, Qy, Qz]
        domain_head (via GRL)       →  [synthetic, real] logits

    Args:
        num_input_channels: Event frame channels (3 for 3-channel decay, 5 for voxel)
        backbone: 'resnet50' (default) or 'resnet18'
        hidden_dim: GRU hidden state size
        gru_layers: Number of stacked GRU layers
        dropout: Dropout probability in pose heads
        pretrained_backbone: Load ImageNet weights for backbone
        grl_alpha: Gradient reversal strength (ramped externally during DANN training)
    """

    def __init__(
        self,
        num_input_channels: int = 3,
        backbone: str = 'resnet50',
        hidden_dim: int = 512,
        gru_layers: int = 1,
        dropout: float = 0.2,
        pretrained_backbone: bool = True,
        grl_alpha: float = 0.0,
    ):
        super().__init__()

        self.num_input_channels = num_input_channels
        self.backbone_name = backbone

        # --- Backbone ---
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            resnet = resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            resnet = resnet18(weights=weights)
            self.feature_dim = 512

        # Patch conv1 for arbitrary input channels (same approach as DirectPoseCNN)
        original_conv1 = resnet.conv1
        if num_input_channels != 3:
            self.conv1 = nn.Conv2d(
                num_input_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                if pretrained_backbone:
                    avg_w = original_conv1.weight.data.mean(dim=1, keepdim=True)  # (64,1,7,7)
                    self.conv1.weight.data = avg_w.repeat(1, num_input_channels, 1, 1)
                else:
                    nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        else:
            self.conv1 = original_conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # --- Temporal memory ---
        self.gru = nn.GRU(
            self.feature_dim, hidden_dim,
            num_layers=gru_layers, batch_first=True
        )

        # --- Pose heads ---
        head_hidden = hidden_dim // 2
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 3)
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 4)
        )

        # --- Domain head (GRL + classifier) ---
        self.domain_grl = GradientReversalLayer(alpha=grl_alpha)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def set_grl_alpha(self, alpha: float):
        """Update GRL strength — call each epoch for standard DANN ramp."""
        self.domain_grl.alpha = alpha

    def forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """CNN backbone: (batch, C, H, W) → (batch, feature_dim)"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, voxel_seq: torch.Tensor):
        """
        Args:
            voxel_seq: (batch, seq_len, C, H, W)

        Returns:
            translation:   (batch, 3)
            rotation:      (batch, 4)  normalized quaternion
            domain_logits: (batch, 2)  raw logits [synthetic, real]
        """
        batch, seq_len, C, H, W = voxel_seq.shape

        # Process ALL frames in one CNN pass — eliminates the Python loop and lets
        # the GPU saturate on one large (batch*seq_len, C, H, W) forward instead of
        # seq_len separate (batch, C, H, W) passes.
        features = self.forward_cnn(
            voxel_seq.reshape(batch * seq_len, C, H, W)
        ).reshape(batch, seq_len, self.feature_dim)  # (batch, seq_len, feature_dim)

        gru_out, _ = self.gru(features)  # (batch, seq_len, hidden_dim)
        last_out = gru_out[:, -1]        # use last timestep only

        trans = self.translation_head(last_out)
        rot = F.normalize(self.rotation_head(last_out), p=2, dim=-1)
        domain_logits = self.domain_head(self.domain_grl(last_out))

        return trans, rot, domain_logits
