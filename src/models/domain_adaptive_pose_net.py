import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from .gradient_reversal import GradientReversalLayer

class DomainAdaptivePoseNet(nn.Module):
    def __init__(self, B, hidden_dim=256, gru_layers=1):
        super().__init__()
        # Backbone: ResNet-18, modify first conv for B channels
        backbone = resnet18(pretrained=True)
        backbone.conv1 = nn.Conv2d(B, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = backbone
        self.feature_dim = backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Temporal Memory: GRU
        self.gru = nn.GRU(self.feature_dim, hidden_dim, num_layers=gru_layers, batch_first=True)

        # Heads
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 3)
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 4)
        )
        self.domain_grl = GradientReversalLayer(alpha=1.0)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, voxel_seq):
        # voxel_seq: (batch, seq_len, B, H, W)
        batch, seq_len, B, H, W = voxel_seq.shape
        features = []
        for t in range(seq_len):
            f = self.backbone(voxel_seq[:, t])
            features.append(f)
        features = torch.stack(features, dim=1)  # (batch, seq_len, feature_dim)
        gru_out, _ = self.gru(features)  # (batch, seq_len, hidden_dim)
        last_out = gru_out[:, -1]  # Use last timestep

        trans = self.translation_head(last_out)
        rot = self.rotation_head(last_out)
        rot = F.normalize(rot, p=2, dim=-1)
        domain_logits = self.domain_head(self.domain_grl(last_out))
        return trans, rot, domain_logits
