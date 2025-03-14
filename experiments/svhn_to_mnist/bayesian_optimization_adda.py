"""
Implements WDGRL with clustering
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""
import torch
import numpy as np
import sys
sys.path.append('/datacommons/carlsonlab/yl407/packages')
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
import argparse
import random
from torch import nn
import math
import pandas as pd
import os
sys.path.insert(0, '../../')
import itertools
from torch.autograd import grad
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torchvision import datasets, transforms
import utils.config as config
from utils.helper import *
from models.model_DANN import *
from data.mnist_mnistm_data import *
from data.svhn import *
from geomloss import SamplesLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from typing import Dict, List, Optional, Tuple

seed_new = 123
seed_torch(seed=seed_new)

batch_size = 128
half_batch = batch_size // 2

##########source data########################
source_dataset = load_SVHN_LS(target_num=1500,train_flag='train')
source_loader = DataLoader(source_dataset, batch_size=half_batch, drop_last=True,\
                           shuffle=True, num_workers=0, pin_memory=True)

##########target data########################
target_dataset_label = load_mnist_LS(source_num=290,train_flag=False)

target_loader = DataLoader(target_dataset_label, batch_size=half_batch,drop_last=True,\
                           shuffle=True, num_workers=0, pin_memory=True)

    
eval_dataset = load_mnist_LS(source_num=290,train_flag=False)

eval_dataloader_all = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False,
                    drop_last=False, num_workers=0, pin_memory=True)


def train(
    Net: torch.nn.Module,\
    source_loader: DataLoader,\
    target_loader: DataLoader,\
    parameters: Dict[str, float],
) -> nn.Module: 
    seed_torch(seed=123)
    
    lr = parameters.get("lr", 0.001) 
    iterations = parameters.get("iterations", 500) 
    epochs = parameters.get("epochs", 2) 
    k_disc = parameters.get("k_disc", 1) 
    k_clf = parameters.get("k_clf", 1)  
    
    source_model = Net().to(device)
    source_model.load_state_dict(torch.load('trained_models/DANN_source.pt'))
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    
    clf = source_model
    source_model = source_model.feature_extractor

    target_model = Net().to(device)
    target_model.load_state_dict(torch.load('trained_models/DANN_source.pt'))
    target_model = target_model.feature_extractor

    discriminator = nn.Sequential(
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)


    discriminator_optim = torch.optim.Adam(discriminator.parameters(),lr=0.001)
    target_optim = torch.optim.Adam(target_model.parameters(),lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    best_target_acc = 0
    for epoch in range(1, epochs+1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        total_loss = 0
        total_accuracy = 0
        for _ in trange(iterations, leave=False):
            # Train discriminator
            set_requires_grad(target_model, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(k_disc):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)

                source_features = source_model(source_x).view(source_x.shape[0], -1)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                             torch.zeros(target_x.shape[0], device=device)])

                preds = discriminator(discriminator_x).squeeze()
                loss = criterion(preds, discriminator_y)

                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()

                total_loss += loss.item()
                total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()
                    # Train classifier
            
            set_requires_grad(target_model, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(k_clf):
                _, (target_x, _) = next(batch_iterator)
                target_x = target_x.to(device)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)

                preds = discriminator(target_features).squeeze()
                loss = criterion(preds, discriminator_y)

                target_optim.zero_grad()
                loss.backward()
                target_optim.step()
            clf.feature_extractor = target_model
    return clf


def evaluate(
    clf: nn.Module, \
    eval_dataloader_all: DataLoader
) -> float:
    """
    Compute classification accuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
    Returns:
        float: classification accuracy
    """
    
    seed_new = 123
    seed_torch(seed=seed_new)
    
    total_accuracy = 0
    with torch.no_grad():
        for x, y_true in tqdm(eval_dataloader_all, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = clf(x)
            total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()

    mean_accuracy = total_accuracy / len(eval_dataloader_all)

    return mean_accuracy

def train_evaluate(parameterization):
    seed_new = 123
    seed_torch(seed=seed_new)
    
    clf = train(Net = Net,\
                                                                  source_loader=source_loader,\
                                                                  target_loader=target_loader,
                                                                  parameters=parameterization)
    return evaluate(
        clf=clf,\
        eval_dataloader_all = eval_dataloader_all
    )

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "k_disc", "type": "choice", "values": [1, 2, 3]},
        {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2]},
        {"name": "k_clf", "type": "choice", "values": [1, 2, 3]},
        #{"name": "lamb_wd", "type": "range", "bounds": [0.0, 1.0]},
        #{"name": "lamb_centroid", "type": "range", "bounds": [0.0, 1.0]},
        #{"name": "lamb_sntg", "type": "range", "bounds": [0.0, 1.0]},
        #{"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
    ],
    evaluation_function=train_evaluate,
    objective_name='accuracy',
)


print("best_parameters",best_parameters)
means, covariances = values
print("means",means)
print("covariances",covariances)

