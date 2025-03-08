import sys
sys.path.insert(0, '../../')
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import math
import numpy as np
from utils.helper import *
from torch.nn.utils import spectral_norm

##Net - feature extractor
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(50*4*4, 128),
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return features


##Net - classifier  
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.Linear(16, 10),
        )
        
    def forward(self, x):
        logits = self.classifier(x)
        return logits
