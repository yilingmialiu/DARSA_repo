import sys
sys.path.append('/datacommons/carlsonlab/yl407/packages')
sys.path.insert(0, '../../')
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import utils.config as config
from torchvision.datasets import MNIST,USPS
from torchvision.transforms import Compose, ToTensor
from utils.helper import *

def load_USPS(train_flag=False):
    target_dataset = USPS('../../data/usps', train=train_flag, download=True,
                          transform=Compose([transforms.Resize(28),
                                             GrayscaleToRgb(), 
                                             ToTensor()]))


    #target_dataset.data = target_dataset.data[indices_target]
    #target_dataset.targets = np.array(target_dataset.targets)[indices_target]
    return target_dataset


def load_USPS_LS(target_num,alpha,train_flag=False):
    target_dataset = USPS('../../data/usps', train=train_flag, download=True,
                          transform=Compose([transforms.Resize(28),
                                             GrayscaleToRgb(), 
                                             ToTensor()]))
    indices_target = []
    
    for index in range(10):
        if index%2 == 0:
            indices_even_temp = [i for i,x in enumerate(target_dataset.targets) if x==index]
            indices_temp_sample = np.random.choice(indices_even_temp,target_num*alpha,replace=False)
        if index%2 != 0:
            indices_odd_temp = [i for i,x in enumerate(target_dataset.targets) if x==index]
            indices_temp_sample = np.random.choice(indices_odd_temp,target_num,replace=False)
        
        indices_target = np.concatenate((indices_target, indices_temp_sample))
        indices_target = indices_target.astype(int)
        np.random.shuffle(indices_target)

    target_dataset.data = target_dataset.data[indices_target]
    target_dataset.targets = np.array(target_dataset.targets)[indices_target]
    return target_dataset


def load_USPS_LS_DRANet(target_num,alpha,train_flag=False):
    target_dataset = USPS('../../data/usps', train=train_flag, download=True,
                          transform=Compose([transforms.Resize(64),
                                             GrayscaleToRgb(), 
                                             ToTensor()]))
    indices_target = []
    
    for index in range(10):
        if index%2 == 0:
            indices_even_temp = [i for i,x in enumerate(target_dataset.targets) if x==index]
            indices_temp_sample = np.random.choice(indices_even_temp,target_num*alpha,replace=False)
        if index%2 != 0:
            indices_odd_temp = [i for i,x in enumerate(target_dataset.targets) if x==index]
            indices_temp_sample = np.random.choice(indices_odd_temp,target_num,replace=False)
        
        indices_target = np.concatenate((indices_target, indices_temp_sample))
        indices_target = indices_target.astype(int)
        np.random.shuffle(indices_target)

    target_dataset.data = target_dataset.data[indices_target]
    target_dataset.targets = np.array(target_dataset.targets)[indices_target]
    return target_dataset


