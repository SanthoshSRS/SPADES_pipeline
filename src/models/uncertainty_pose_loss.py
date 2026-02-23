import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma_t = nn.Parameter(torch.tensor(0.0))
        self.sigma_r = nn.Parameter(torch.tensor(0.0))

    def rotation_geodesic_loss(self, q_pred, q_true):
        # q_pred, q_true: (batch, 4), normalized
        dot = torch.abs(torch.sum(q_pred * q_true, dim=-1))
        dot = torch.clamp(dot, -1.0, 1.0)
        return 2 * torch.acos(dot)

    def forward(self, t_pred, t_true, q_pred, q_true):
        mse_t = F.mse_loss(t_pred, t_true)
        geo_r = torch.mean(self.rotation_geodesic_loss(q_pred, q_true))
        loss = (
            torch.exp(-self.sigma_t) * mse_t + self.sigma_t +
            torch.exp(-self.sigma_r) * geo_r + self.sigma_r
        )
        return loss
