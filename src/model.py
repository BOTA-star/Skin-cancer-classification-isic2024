import torch.nn as nn
from torchvision import models

def get_model(pretrained=True):
    model = models.resnet18(pretrained=pretrained)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)

    return model