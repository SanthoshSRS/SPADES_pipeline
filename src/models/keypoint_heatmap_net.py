"""
KeypointHeatmapNet: ResNet-50 backbone + heatmap decoder for satellite keypoint localization.

Replaces GlobalAvgPool → FC regression with a spatial heatmap decoder, preserving
the spatial feature map through the full forward pass.  Per-keypoint Gaussian heatmaps
are trained with MSE loss; at inference, soft-argmax extracts sub-pixel (u,v) coords
which are fed to cv2.solvePnPRansac for 6-DoF pose.

Architecture (SimpleBaseline-style, Xiao et al. ECCV 2018):
    ResNet-50 (no avgpool)  →  layer4: (B, 2048, H/32, W/32)
    ConvTranspose2d(2048→256, k=4, s=2, p=1)  → (B, 256, H/16, W/16) + BN + ReLU
    ConvTranspose2d(256→256,  k=4, s=2, p=1)  → (B, 256, H/8,  W/8)  + BN + ReLU
    ConvTranspose2d(256→256,  k=4, s=2, p=1)  → (B, 256, H/4,  W/4)  + BN + ReLU
    Conv2d(256→K, k=1)                         → (B, K,   H/4,  W/4) heatmaps

At inference: soft_argmax_2d(heatmaps) → (B, K, 2) normalized [0, 1] coords.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ─────────────────────────────────────────────────────────────────────────────
# Functional utilities (used by training script and inference)
# ─────────────────────────────────────────────────────────────────────────────

def soft_argmax_2d(heatmaps: torch.Tensor, temperature: float = 100.0) -> torch.Tensor:
    """
    Differentiable soft-argmax over 2D heatmaps.

    Computes the expected (u, v) location under a softmax distribution over
    the spatial dimensions.  Temperature > 1 sharpens the distribution toward
    the hard argmax; temperature = 1 is standard softmax.

    Args:
        heatmaps:    (B, K, H, W) raw logits (no activation applied yet)
        temperature: Scale factor before softmax (default: 100)

    Returns:
        coords: (B, K, 2) normalized [0, 1] — (u, v) = (horizontal, vertical)
    """
    B, K, H, W = heatmaps.shape
    # Work in float32 to avoid float16 saturation at high temperature
    hm = heatmaps.float()
    weights = F.softmax(hm.reshape(B, K, -1) * temperature, dim=-1)  # (B, K, H*W)
    weights = weights.reshape(B, K, H, W)

    # Coordinate grids in [0, 1]
    xx = torch.linspace(0.0, 1.0, W, device=heatmaps.device)
    yy = torch.linspace(0.0, 1.0, H, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')   # (H, W)

    u = (weights * grid_x).sum(dim=(-2, -1))   # (B, K) horizontal
    v = (weights * grid_y).sum(dim=(-2, -1))   # (B, K) vertical
    return torch.stack([u, v], dim=-1)          # (B, K, 2) in [0, 1]


def make_gaussian_heatmaps(
    kp_norm: torch.Tensor,
    H_hm: int,
    W_hm: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    """
    Generate 2D Gaussian target heatmaps from normalized keypoint coordinates.

    Args:
        kp_norm: (B, K, 2) normalized [0, 1] coords (u horizontal, v vertical)
        H_hm:    Heatmap height in pixels
        W_hm:    Heatmap width in pixels
        sigma:   Gaussian sigma in heatmap pixel units (default: 2.0)

    Returns:
        targets: (B, K, H_hm, W_hm) float32 Gaussians centred at each keypoint
    """
    B, K, _ = kp_norm.shape
    device = kp_norm.device

    yy = torch.arange(H_hm, device=device, dtype=torch.float32)   # (H_hm,)
    xx = torch.arange(W_hm, device=device, dtype=torch.float32)   # (W_hm,)

    # Convert normalized [0,1] → heatmap pixel space
    u_hm = kp_norm[..., 0].float() * (W_hm - 1)   # (B, K)
    v_hm = kp_norm[..., 1].float() * (H_hm - 1)   # (B, K)

    # Squared distance — broadcast over (B, K, H_hm, W_hm)
    dy = yy.view(1, 1, H_hm, 1) - v_hm.view(B, K, 1, 1)
    dx = xx.view(1, 1, 1, W_hm) - u_hm.view(B, K, 1, 1)

    return torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))   # (B, K, H_hm, W_hm)


def heatmap_loss(
    pred_hm: torch.Tensor,
    kp_norm: torch.Tensor,
    vis: torch.Tensor,
    sigma: float = 2.0,
) -> torch.Tensor:
    """
    MSE loss between predicted heatmaps and Gaussian targets, masked to visible keypoints.

    Args:
        pred_hm: (B, K, H_hm, W_hm)  raw model output (no sigmoid / activation)
        kp_norm: (B, K, 2)            normalized [0, 1] ground truth coords
        vis:     (B, K)               bool — True for visible keypoints
        sigma:   Gaussian sigma in heatmap pixels (default: 2.0)

    Returns:
        scalar MSE loss averaged over visible keypoints and heatmap pixels
    """
    H_hm, W_hm = pred_hm.shape[2], pred_hm.shape[3]
    gt_hm = make_gaussian_heatmaps(kp_norm, H_hm, W_hm, sigma)   # (B, K, H_hm, W_hm) float32

    # Cast prediction to float32 so MSE is numerically stable under AMP
    pred_f = pred_hm.float()

    # Mask: only penalize visible keypoints
    mask = vis.float().unsqueeze(-1).unsqueeze(-1)   # (B, K, 1, 1)
    n_vis = vis.float().sum().clamp(min=1.0)

    loss = ((pred_f - gt_hm) ** 2 * mask).sum() / (n_vis * H_hm * W_hm)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class KeypointHeatmapNet(nn.Module):
    """
    ResNet-50 + transposed-convolution heatmap decoder for satellite keypoint prediction.

    Preserves the spatial feature map from ResNet-50 layer4 (H/32 × W/32) and
    upsamples through 3× ConvTranspose2d layers to produce K heatmaps at H/4 × W/4.

    Args:
        num_keypoints:       Number of keypoints K (default: 8)
        num_input_channels:  Input channels — 3 for ThreeChannelEventFrame
        pretrained_backbone: Load ImageNet weights for conv layers (default: True)
        dropout:             Dropout probability (unused in decoder — kept for API parity)
    """

    IMG_W = 1280
    IMG_H = 720

    def __init__(
        self,
        num_keypoints: int = 8,
        num_input_channels: int = 3,
        pretrained_backbone: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_input_channels = num_input_channels

        # ── ResNet-50 backbone (same channel-patching as KeypointPoseNet) ────
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
            self.conv1 = resnet.conv1   # reuse pretrained RGB weights directly

        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        # Note: resnet.avgpool is intentionally NOT added — we keep spatial dims.

        # ── Heatmap decoder: (2048, H/32) → (K, H/4) via 3× ConvTranspose ───
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, kernel_size=1),   # K heatmap logits
        )

        # Initialise decoder weights (backbone already initialised from ImageNet)
        for m in self.decoder.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """ResNet-50 spatial features.  Returns (B, 2048, H/32, W/32)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x   # (B, 2048, H/32, W/32) — NO avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) or (B, 1, C, H, W) event frame

        Returns:
            heatmaps: (B, K, H/4, W/4) raw logits — feed to heatmap_loss for training,
                      or soft_argmax_2d for inference coordinates
        """
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)   # (B, 1, C, H, W) → (B, C, H, W)
        feats = self.forward_features(x)      # (B, 2048, H/32, W/32)
        return self.decoder(feats)            # (B, K, H/4, W/4)

    def predict_coords(self, x: torch.Tensor, temperature: float = 100.0) -> torch.Tensor:
        """
        End-to-end: image → normalized (u,v) keypoint coordinates.

        Args:
            x:           (B, C, H, W) event frame
            temperature: Soft-argmax temperature (default: 100)

        Returns:
            coords: (B, K, 2) normalized [0, 1] — multiply by [IMG_W, IMG_H] for pixels
        """
        heatmaps = self.forward(x)
        return soft_argmax_2d(heatmaps, temperature)
