import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class VanillaModel(nn.Module):
    def __init__(self):
        super(VanillaModel, self).__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # Consistent naming for the unfreeze logic
        self.spatial_features = base.features
        self.avgpool = base.avgpool
        
        # Binary Classifier Head
        in_features = base.classifier[3].in_features
        self.classifier = base.classifier
        self.classifier[3] = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.spatial_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model():
    return VanillaModel()