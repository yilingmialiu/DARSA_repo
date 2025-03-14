import sys
sys.path.append('/datacommons/carlsonlab/yl407/packages')
sys.path.insert(0, '../../')

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import utils.config as config
from torchvision.datasets import MNIST,USPS,SVHN
from torchvision.transforms import Compose, ToTensor
from utils.helper import *

def load_SVHN(train_flag='train'):
    target_dataset = SVHN('../../data/svhn', split=train_flag, download=True,
                          transform=Compose([transforms.Resize(28),
                                             ToTensor()]))
    return target_dataset


def load_SVHN_LS(target_num,alpha,train_flag='train'):
    target_dataset = SVHN('../../data/svhn', split=train_flag, download=True,
                          transform=Compose([transforms.Resize(28),
                                             ToTensor()]))
    indices_target = []
    
    for index in range(10):
        if index%2 == 0:
            indices_even_temp = [i for i,x in enumerate(target_dataset.labels) if x==index]
            indices_temp_sample = np.random.choice(indices_even_temp,target_num*alpha,replace=False)
        if index%2 != 0:
            indices_odd_temp = [i for i,x in enumerate(target_dataset.labels) if x==index]
            indices_temp_sample = np.random.choice(indices_odd_temp,target_num,replace=False)
        
        indices_target = np.concatenate((indices_target, indices_temp_sample))
        indices_target = indices_target.astype(int)
        np.random.shuffle(indices_target)

    target_dataset.data = target_dataset.data[indices_target]
    target_dataset.labels = np.array(target_dataset.labels)[indices_target]
    return target_dataset

def load_SVHN_LS_DRANet(target_num,alpha,train_flag='train'):
    target_dataset = SVHN('../../data/svhn', split=train_flag, download=True,
                          transform=Compose([transforms.Resize(64),
                                             ToTensor()]))
    indices_target = []
    
    for index in range(10):
        if index%2 == 0:
            indices_even_temp = [i for i,x in enumerate(target_dataset.labels) if x==index]
            indices_temp_sample = np.random.choice(indices_even_temp,target_num*alpha,replace=False)
        if index%2 != 0:
            indices_odd_temp = [i for i,x in enumerate(target_dataset.labels) if x==index]
            indices_temp_sample = np.random.choice(indices_odd_temp,target_num,replace=False)
        
        indices_target = np.concatenate((indices_target, indices_temp_sample))
        indices_target = indices_target.astype(int)
        np.random.shuffle(indices_target)

    target_dataset.data = target_dataset.data[indices_target]
    target_dataset.labels = np.array(target_dataset.labels)[indices_target]
    return target_dataset

def load_SVHN_DRANet(train_flag='train'):
    target_dataset = SVHN('../../data/svhn', split=train_flag, download=True,
                          transform=Compose([transforms.Resize(64),
                                             ToTensor()]))
    return target_dataset