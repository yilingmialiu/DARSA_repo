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

##DenseNet - feature extractor for neural data
class DenseNet(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(dim_in, 256), 
            nn.ReLU(), 
            #nn.Dropout(),
            nn.Linear(256, 128), 
            nn.ReLU(),
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return features

    
##Net - classifier  
class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.Linear(16, num_classes),
        )
        
    def forward(self, x):
        logits = self.classifier(x)
        return logits


class DenseNet_bw(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(dim_in, 256), 
            nn.LeakyReLU(), 
            #nn.ReLU(), 
            #nn.Dropout(),
            nn.Linear(256, 128), 
            nn.Softplus(), 
            nn.ReLU(),
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return features