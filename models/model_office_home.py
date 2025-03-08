import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Function

# Define model_urls manually since it's no longer in torchvision
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReverseLayer(nn.Module):
    def __init__(self, alpha):
        super(GradientReverseLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.alpha)

class Classifier(nn.Module):
    """Classifier for VisDA-2017"""
    def __init__(self, num_classes=12, feature_dim=256):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        return self.layer(x)

class Net(nn.Module):
    """Basic Network"""

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=256, width=None, pool_layer=None, finetune=True):
        super(Net, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Get input features dimension
        if hasattr(backbone, 'out_features'):
            input_dim = backbone.out_features
        elif hasattr(backbone, 'hidden_dim'):  # For ViT
            input_dim = backbone.hidden_dim
        else:
            input_dim = 768  # Default for ViT base

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, num_classes)
        )
        self.pool_layer = pool_layer if pool_layer else nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        if not finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, tuple):  # Some models return tuple
            features = features[0]
        features = self.pool_layer(features)
        bottleneck = self.bottleneck(features)
        return bottleneck

def get_model(model_name, pretrain=True):
    """Get backbone model with pretrained weights if specified"""
    if model_name == 'vit_base_patch16_224':
        if pretrain:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            backbone = models.vit_b_16(weights=weights)
        else:
            backbone = models.vit_b_16()
        backbone.heads = nn.Identity()  # Remove classification head
        # Add out_features attribute for ViT
        backbone.out_features = backbone.hidden_dim  # 768 for base model
        return backbone
    else:
        raise ValueError(f"Model {model_name} not supported")