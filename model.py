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
    backbone_weights_path: str = None,
):
    """
    Build image backbone.
    Local CPU demo nên dùng mobilenet_v3_small.
    Train thật GPU có thể dùng efficientnet_b0 hoặc efficientnet_b3.
    """

    backbone_name = backbone_name.lower()
    model = None

    if backbone_name == "mobilenet_v3_small":
        if pretrained and not backbone_weights_path:
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

    elif backbone_name == "efficientnet_b0":
        if pretrained and not backbone_weights_path:
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

    elif backbone_name == "efficientnet_b3":
        if pretrained and not backbone_weights_path:
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

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if backbone_weights_path:
        import os
        if os.path.exists(backbone_weights_path):
            print(f"[INFO] Loading offline backbone weights from: {backbone_weights_path}")
            try:
                state_dict = torch.load(backbone_weights_path, map_location="cpu")
                # Xử lý trường hợp load toàn bộ object checkpoint
                if isinstance(state_dict, dict):
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    elif "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    elif "model" in state_dict:
                        state_dict = state_dict["model"]
                
                # Loại bỏ prefix nếu có
                normalized = {}
                for k, v in state_dict.items():
                    key = k
                    if key.startswith("module."):
                        key = key[7:]
                    if key.startswith("features."):
                        # PyTorch torchvision models usually have features. prefix natively, 
                        # so we shouldn't strip it unless it mismatches.
                        pass
                    normalized[key] = v
                
                missing, unexpected = model.load_state_dict(normalized, strict=False)
                print(f"[OK] Backbone loaded. Missing: {len(missing)} | Unexpected: {len(unexpected)}")
            except Exception as e:
                print(f"[ERROR] Failed to load offline backbone weights: {e}")
        else:
            print(f"[WARNING] Offline backbone weights not found: {backbone_weights_path}")

    return model, image_feature_dim

class ISICMultimodalModel(nn.Module):
    def __init__(
        self,
        meta_dim: int,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = False,
        backbone_weights_path: str = None,
        metadata_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.image_model, image_feature_dim = build_image_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            backbone_weights_path=backbone_weights_path,
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