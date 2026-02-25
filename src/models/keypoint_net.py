"""
KeypointPoseNet: ResNet-50 → 8 keypoint (u,v) predictions for PnP pose solving.

Architecture:
    ResNet-50 backbone (same as DirectPoseCNN, 3-channel input) → 2048-d features
    Keypoint head: FC(2048 → 512 → ReLU → Dropout → 16) → Sigmoid
    Output: 8 × (u_norm, v_norm) in [0, 1]

At inference, multiply by [IMG_W, IMG_H] = [1280, 720] to get pixel coordinates,
then feed to cv2.solvePnP with the known 3D body-frame keypoints.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class KeypointPoseNet(nn.Module):
    """
    Predicts 8 satellite keypoint positions for PnP-based pose estimation.

    Output: (B, 16) — 8 × (u_norm, v_norm) normalized to [0, 1].
    Multiply by [1280, 720] for pixel coordinates before PnP.

    Args:
        num_input_channels: Input channels (3 for ThreeChannelEventFrame)
        dropout:            Dropout in keypoint head (default: 0.2)
        pretrained_backbone: Use ImageNet pretrained ResNet-50 (default: True)
    """

    IMG_W        = 1280
    IMG_H        = 720
    NUM_KEYPOINTS = 8

    def __init__(
        self,
        num_input_channels: int = 3,
        dropout: float = 0.2,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        self.num_input_channels = num_input_channels

        # ── ResNet-50 backbone (identical to DirectPoseCNN) ──────────────────
        if pretrained_backbone:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet50(weights=None)

        if num_input_channels != 3:
            original_conv1 = resnet.conv1
            self.conv1 = nn.Conv2d(
                num_input_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                if pretrained_backbone:
                    avg_w = original_conv1.weight.data.mean(dim=1, keepdim=True)
                    self.conv1.weight.data = avg_w.repeat(1, num_input_channels, 1, 1)
                else:
                    nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out',
                                            nonlinearity='relu')
        else:
            self.conv1 = resnet.conv1  # reuse pretrained RGB weights directly

        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool

        # ── Keypoint head: 2048 → 512 → 16 (normalized [0,1]) ───────────────
        self.kp_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, self.NUM_KEYPOINTS * 2),
            nn.Sigmoid(),    # clamp output to [0, 1]
        )

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """ResNet-50 feature extraction. Returns (B, 2048)."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)  or  (B, 1, C, H, W) — event frame(s)

        Returns:
            kp: (B, 16) — 8 × (u_norm, v_norm) in [0, 1]
        """
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)     # (B, 1, C, H, W) → (B, C, H, W)
        return self.kp_head(self.forward_backbone(x))
