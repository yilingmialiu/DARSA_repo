import sys
sys.path.insert(0, '../../')
sys.path.append('/datacommons/carlsonlab/yl407/packages')
import torch.nn as nn
from torchvision import models
#from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls
import copy
from models.model_office_home_classifier import Classifier as ClassifierBase
from typing import Tuple, Optional, List, Dict


class Net_resnset(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(Net_resnset, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)


class Net(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            #nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            #nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(Net, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)


class Classifier(nn.Module):
    def __init__(self,input_dim: Optional[int] = 256):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 65),
        )
        
    def forward(self, x):
        logits = self.classifier(x)
        return logits