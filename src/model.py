import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    efficientnet_b3,
    mobilenet_v3_small,
)

def build_image_backbone(
    backbone_name: str = "mobilenet_v3_small",
    pretrained: bool = False,
):
    """
    Build image backbone.
    Local CPU demo nên dùng mobilenet_v3_small.
    Train thật GPU có thể dùng efficientnet_b0 hoặc efficientnet_b3.
    """

    backbone_name = backbone_name.lower()

    if backbone_name == "mobilenet_v3_small":
        if pretrained:
            try:
                from torchvision.models import MobileNet_V3_Small_Weights
                model = mobilenet_v3_small(
                    weights=MobileNet_V3_Small_Weights.DEFAULT
                )
            except Exception as e:
                print("[WARNING] Cannot load pretrained MobileNetV3 weights.")
                print("Reason:", e)
                model = mobilenet_v3_small(weights=None)
        else:
            model = mobilenet_v3_small(weights=None)

        image_feature_dim = model.classifier[0].in_features
        model.classifier = nn.Identity()

        return model, image_feature_dim

    if backbone_name == "efficientnet_b0":
        if pretrained:
            try:
                from torchvision.models import EfficientNet_B0_Weights
                model = efficientnet_b0(
                    weights=EfficientNet_B0_Weights.DEFAULT
                )
            except Exception as e:
                print("[WARNING] Cannot load pretrained EfficientNet-B0 weights.")
                print("Reason:", e)
                model = efficientnet_b0(weights=None)
        else:
            model = efficientnet_b0(weights=None)

        image_feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()

        return model, image_feature_dim

    if backbone_name == "efficientnet_b3":
        if pretrained:
            try:
                from torchvision.models import EfficientNet_B3_Weights
                model = efficientnet_b3(
                    weights=EfficientNet_B3_Weights.DEFAULT
                )
            except Exception as e:
                print("[WARNING] Cannot load pretrained EfficientNet-B3 weights.")
                print("Reason:", e)
                model = efficientnet_b3(weights=None)
        else:
            model = efficientnet_b3(weights=None)

        image_feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()

        return model, image_feature_dim

    raise ValueError(f"Unsupported backbone: {backbone_name}")

class ISICMultimodalModel(nn.Module):
    def __init__(
        self,
        meta_dim: int,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = False,
        metadata_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.image_model, image_feature_dim = build_image_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )

        self.meta_model = nn.Sequential(
            nn.Linear(meta_dim, metadata_hidden_dim),
            nn.BatchNorm1d(metadata_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(metadata_hidden_dim, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + 32, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, images, metas):
        image_features = self.image_model(images)
        meta_features = self.meta_model(metas)

        combined_features = torch.cat(
            [image_features, meta_features],
            dim=1,
        )

        logits = self.classifier(combined_features)

        return logits

    def freeze_image_backbone(self):
        for param in self.image_model.parameters():
            param.requires_grad = False

    def unfreeze_image_backbone(self):
        for param in self.image_model.parameters():
            param.requires_grad = True