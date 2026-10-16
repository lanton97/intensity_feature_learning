from modules.model import XFeatModel, BasicLayer
from torch import nn
import torch
from torch.nn import functional as F

class IntensityFeatureModel(XFeatModel):
    def __init__(self):
        super().__init__()
        self.mu_head = nn.Sequential(
            BasicLayer(65, 64, 1, padding=0),
            nn.Conv2d(64, 2, 1),
            nn.Sigmoid()
        )
        self.cholesky_head = nn.Sequential(
            BasicLayer(65, 64, 1, padding=0),
            nn.Conv2d(64, 3, 1),
        )

    def forward(self, x):
        feats, kp_logits, heatmap = super().forward(x)
        # Compute mu and log_std for keypoints
        # Essentially fitting a gaussian over the keypoint x,y coordinates in 8x8 grid 
        mu = self.mu_head(kp_logits) * 8.0              # (B, 2, Hc, Wc)
        c  = self.cholesky_head(kp_logits)              # (B, 3, Hc, Wc)
        B, _, Hc, Wc = c.shape
        
        L = c.new_zeros(B, Hc, Wc, 2, 2)
        # Apply softplus to ensure positive diagonal entries for Cholesky factor
        L[..., 0, 0] = F.softplus(c[:, 0]) + 1e-3       # L11 > 0
        L[..., 1, 0] = c[:, 1]                          # L21 unconstrained
        L[..., 1, 1] = F.softplus(c[:, 2]) + 1e-3       # L22 > 0
        cov = L @ L.transpose(-1, -2)                   # (B, Hc, Wc, 2, 2)
        
        mu = mu.permute(0, 2, 3, 1)                     # (B, Hc, Wc, 2) to match L
        return feats, kp_logits, heatmap, mu, L, cov
