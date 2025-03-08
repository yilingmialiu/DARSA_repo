import sys
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
import pickle as pkl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def load_neural_b_to_w(num_b=[6000,3000,6000],num_w = [3000,6000,3000]):
    trainDict = pkl.load(open(os.path.join(dataDir,'tst_train_dict.pkl'),'rb'))
    testDict = pkl.load(open(os.path.join(dataDir,'tst_test_dict.pkl'),'rb'))
    X = np.concatenate([trainDict['X_psd'],testDict['X_psd']])
    y_geno = np.concatenate([trainDict['y_geno'],testDict['y_geno']])
    y_task = np.concatenate([trainDict['y_task'],testDict['y_task']])
    X_source = X[np.where(y_geno==True)]
    X_target = X[np.where(y_geno==False)]
    _, y_source = np.unique(y_task[np.where(y_geno==True)], return_inverse=True)
    _, y_target = np.unique(y_task[np.where(y_geno==False)], return_inverse=True)
    
    #create label shifting
    indices_source_0 = np.random.choice(np.where((y_source == 0))[0],num_b[0],replace=False)
    indices_source_1 = np.random.choice(np.where((y_source == 1))[0],num_b[1],replace=False)
    indices_source_2 = np.random.choice(np.where((y_source == 2))[0],num_b[2],replace=False)
    indices_source = np.concatenate((indices_source_0,indices_source_1,indices_source_2))
    np.random.shuffle(indices_source)
    X_source = X_source[indices_source]
    y_source = y_source[indices_source]

    indices_target_0 = np.random.choice(np.where((y_target == 0))[0],num_w[0],replace=False)
    indices_target_1 = np.random.choice(np.where((y_target == 1))[0],num_w[1],replace=False)
    indices_target_2 = np.random.choice(np.where((y_target == 2))[0],num_w[2],replace=False)
    indices_target = np.concatenate((indices_target_0,indices_target_1,indices_target_2))
    np.random.shuffle(indices_target)
    X_target = X_target[indices_target]
    y_target = y_target[indices_target]
    return X_source, y_source, X_target, y_target



def load_neural_w_to_b(num_b=[6000,3000,6000],num_w = [3000,6000,3000]):
    trainDict = pkl.load(open(os.path.join(dataDir,'tst_train_dict.pkl'),'rb'))
    testDict = pkl.load(open(os.path.join(dataDir,'tst_test_dict.pkl'),'rb'))
    X = np.concatenate([trainDict['X_psd'],testDict['X_psd']])
    y_geno = np.concatenate([trainDict['y_geno'],testDict['y_geno']])
    y_task = np.concatenate([trainDict['y_task'],testDict['y_task']])
    X_source = X[np.where(y_geno==False)]
    X_target = X[np.where(y_geno==True)]
    _, y_source = np.unique(y_task[np.where(y_geno==False)], return_inverse=True)
    _, y_target = np.unique(y_task[np.where(y_geno==True)], return_inverse=True)
    
    #create label shifting
    indices_source_0 = np.random.choice(np.where((y_source == 0))[0],num_w[0],replace=False)
    indices_source_1 = np.random.choice(np.where((y_source == 1))[0],num_w[1],replace=False)
    indices_source_2 = np.random.choice(np.where((y_source == 2))[0],num_w[2],replace=False)
    indices_source = np.concatenate((indices_source_0,indices_source_1,indices_source_2))
    np.random.shuffle(indices_source)
    X_source = X_source[indices_source]
    y_source = y_source[indices_source]

    indices_target_0 = np.random.choice(np.where((y_target == 0))[0],num_b[0],replace=False)
    indices_target_1 = np.random.choice(np.where((y_target == 1))[0],num_b[1],replace=False)
    indices_target_2 = np.random.choice(np.where((y_target == 2))[0],num_b[2],replace=False)
    indices_target = np.concatenate((indices_target_0,indices_target_1,indices_target_2))
    np.random.shuffle(indices_target)
    X_target = X_target[indices_target]
    y_target = y_target[indices_target]
    return X_source, y_source, X_target, y_target
