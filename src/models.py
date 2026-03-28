import timm
import torchvision.models as models
import torch.nn as nn

def get_cnn(num_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_vit(num_classes=10):
    model = timm.create_model('vit_tiny_patch16_224', weights='IMAGENET1K_V1')
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model