import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class PhysiologicalBranch(nn.Module):
    """
    A lightweight, rPPG-inspired branch to analyze physiological color inconsistencies.
    Deepfakes often fail to replicate subsurface hemoglobin scattering.
    This branch uses standard convolutions to extract a 32-dim feature vector
    from the color/frequency domain, acting as a "liveness" check.
    """
    def __init__(self, in_channels=3, out_dim=32):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x):
        x = self.conv_block(x)          # (B, 32, 1, 1)
        x = torch.flatten(x, 1)         # (B, 32)
        x = self.fc(x)                  # (B, out_dim)
        return x


class RPPGModel(nn.Module):
    """
    Two-Stream MobileNetV3 + Physiological Branch for lightweight deepfake detection.
    Fuses spatial features (576 dim) with physiological features (32 dim).
    """
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # --- Stream 1: Spatial Backbone ---
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.spatial_features = base.features    # Outputs (B, 576, 7, 7)
        self.avgpool = base.avgpool              # Outputs (B, 576, 1, 1)
        
        # --- Stream 2: Physiological Branch ---
        self.physio_branch = PhysiologicalBranch(in_channels=3, out_dim=32)
        
        # --- Fusion & Classification ---
        # MobileNetV3 spatial (576) + Physio branch (32) = 608 features
        fused_dim = 576 + 32
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        # Stream 1: Spatial
        spatial = self.spatial_features(x)
        spatial = self.avgpool(spatial)
        spatial = torch.flatten(spatial, 1)      # (B, 576)
        
        # Stream 2: Physiological
        physio = self.physio_branch(x)           # (B, 32)
        
        # Fusion
        fused = torch.cat((spatial, physio), dim=1) # (B, 608)
        
        # Classification
        out = self.classifier(fused)
        return out


def get_model(dropout_rate=0.2):
    return RPPGModel(dropout_rate=dropout_rate)