import torch
import torch.nn as nn
from torchvision import models


class ImageMetadataRegressor(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v3_small(weights=weights)
            in_features = net.classifier[0].in_features
            net.classifier = nn.Identity()
            self.encoder = net

        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            in_features = net.classifier[1].in_features
            net.classifier = nn.Identity()
            self.encoder = net

        elif backbone == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b3(weights=weights)
            in_features = net.classifier[1].in_features
            net.classifier = nn.Identity()
            self.encoder = net

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_outputs),
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)
