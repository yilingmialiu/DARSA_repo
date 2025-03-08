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

#Net - feature extractor
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
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

    