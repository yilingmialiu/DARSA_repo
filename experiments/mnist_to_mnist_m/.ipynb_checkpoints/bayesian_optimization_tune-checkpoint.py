"""
Implements WDGRL with clustering
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""
import sys
sys.path.insert(0, '../../')
sys.path.append('/datacommons/carlsonlab/yl407/packages')
import torch
import numpy as np
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
import argparse
import random
import torch
import numpy as np
from torch import nn
import math
import pandas as pd
import os
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
from data.mnist_mnistm_data import *
from torch.nn.utils import spectral_norm
from models.model_mnist_mnistm import *
import utils.config as config
from utils.helper import *
from geomloss import SamplesLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from typing import Dict, List, Optional, Tuple

seed_new = 123
seed_torch(seed=seed_new)

batch_size = 1024
half_batch = batch_size // 2

##########source data########################
source_dataset = load_mnist_LS(source_num=600,alpha=8,train_flag=True)
source_loader = DataLoader(source_dataset, batch_size=half_batch, drop_last=True,\
                           shuffle=True, num_workers=0, pin_memory=True)

##########target data########################
target_dataset_label = load_mnistm_LS(target_num=100,alpha=8,MNISTM=MNISTM,train_flag=True)

target_loader = DataLoader(target_dataset_label, batch_size=half_batch,drop_last=True,\
                           shuffle=True, num_workers=0, pin_memory=True)

    
eval_dataset = load_mnistm_LS(target_num=10,alpha=8,MNISTM=MNISTM,train_flag=False) #target_dataset_label

eval_dataloader_all = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False,
                    drop_last=False, num_workers=0, pin_memory=True)


def train(
    Net: torch.nn.Module,\
    Classifier:torch.nn.Module,\
    source_loader: DataLoader,\
    target_loader: DataLoader,\
    parameters: Dict[str, float],
) -> nn.Module: 
    weight_update = 1
    seed_new = 123
    seed_torch(seed=seed_new)
    
    K = parameters.get("K", 10)
    lr = parameters.get("lr", 0.001) 
    lamb_clf = parameters.get("lamb_clf", 0.9) 
    lamb_wd = parameters.get("lamb_wd", 0.4) 
    lamb_centroid = parameters.get("lamb_centroid", 1) 
    lamb_sntg = parameters.get("lamb_sntg", 1)
    LAMBDA = parameters.get("LAMBDA", 30) 
    momentum = parameters.get("momentum", 0.9)  
    epochs= parameters.get("epochs", 10)  
    
    ##initialize models
    model_feature_source = Net().to(device)
    model_feature_target = Net().to(device)
    model_clf = Classifier().to(device)
    path_to_model = 'trained_models/'
    model_feature_source.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_alpha_'+str(8)+'.pt'))
    model_feature_target.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_alpha_'+str(8)+'.pt'))
    model_clf.load_state_dict(torch.load(path_to_model+'source_clf_rd_128_SGD_alpha_'+str(8)+'.pt'))
    
    model_feature_source = model_feature_source.train()
    model_feature_target = model_feature_target.train()
    model_clf = model_clf.train()
    
    ##initialize model optimizers
    feature_source_optim = torch.optim.SGD(model_feature_source.parameters(), lr=lr,momentum=momentum) 
    feature_target_optim = torch.optim.SGD(model_feature_target.parameters(), lr=lr,momentum=momentum) 
    clf_optim = torch.optim.SGD(model_clf.parameters(), lr=lr,momentum=momentum) 
    
    clf_criterion = nn.CrossEntropyLoss(reduction='none')
    clf_criterion_unweight = nn.CrossEntropyLoss()
    w1_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.05)
    soft_f = nn.Softmax(dim=1)

    ##initialize clustering weights
    w_src = torch.ones(K,1,requires_grad=False).divide(K).to(device)
    w_tgt = torch.ones(K,1,requires_grad=False).divide(K).to(device)
    w_imp = torch.ones(K,1,requires_grad=False).to(device)
            
    for epoch in range(1, epochs+1):
        source_batch_iterator = iter(source_loader)
        target_batch_iterator = iter(target_loader)
        len_dataloader = min(len(source_loader), len(target_loader))
        
        total_unweight_clf_loss = 0
        total_clf_loss = 0
        total_centroid_loss = 0
        total_sntg_loss = 0
        total_w1_loss = 0


        for i in range(len_dataloader):
            
            data_source = next(source_batch_iterator)
            source_x, source_y = data_source
            data_target = next(target_batch_iterator)
            target_x, _ = data_target
            source_x, target_x = source_x.to(device), target_x.to(device)
                      
            set_requires_grad(model_feature_source, requires_grad=True)
            set_requires_grad(model_feature_target, requires_grad=True)
            set_requires_grad(model_clf, requires_grad=True)
            
            ##extract latent features
            source_y = source_y.to(torch.int64).to(device)
            source_feature = model_feature_source(source_x)
            target_feature = model_feature_target(target_x)
            target_feature_2 = model_feature_target(target_x)

            ##unweighted classification loss
            source_preds = model_clf(source_feature)
            clf_loss_unweight = clf_criterion(source_preds, source_y)
            report_clf_loss_unweight = clf_criterion_unweight(source_preds, source_y)
            target_preds = torch.argmax(model_clf(target_feature_2),1)

            ##get clustering information
            source_y = source_y.to(torch.int64).to(device)
            cluster_s = F.one_hot(source_y, num_classes=K).float()
            target_y = target_preds.to(torch.int64).to(device)
            cluster_t = F.one_hot(target_y, num_classes=K).float()

            ##weighted classification loss
            weighted_clf_err = cluster_s * clf_loss_unweight.reshape(-1,1)
            expected_clf_err = torch.mean(weighted_clf_err, dim=0)
            clf_loss = torch.sum(expected_clf_err.reshape(K,1) * w_imp)

            ##weighted domain invariant loss   
            wasserstein_distance = 0
            for cluster_id in range(K):
                if torch.sum(target_preds==cluster_id)!=0 and torch.sum(source_y==cluster_id)!=0:
                    wasserstein_distance += w_tgt[cluster_id]*w1_loss(source_feature[source_y==cluster_id,],\
                                             target_feature[target_preds==cluster_id,]) 
   
            ##clustering loss
            #L_orthogonal
            source_sntg_loss = sntg_loss_func(cluster=cluster_s,feature=source_feature,LAMBDA=LAMBDA)
            target_sntg_loss = sntg_loss_func(cluster=cluster_t,feature=target_feature,LAMBDA=LAMBDA)

            ##calculate centroids
            centroid_loss = centroid_loss_func(K,device,source_y,target_y,source_feature,target_feature)
            sntg_loss = source_sntg_loss + target_sntg_loss
            loss = lamb_clf*clf_loss + lamb_wd * wasserstein_distance + lamb_centroid*centroid_loss + lamb_sntg*sntg_loss 

            #update weights
            with torch.no_grad():
                w_src_batch = cluster_s.mean(dim=0) 
                w_tgt_batch = cluster_t.mean(dim=0)
                w_src = w_src * (1 - weight_update) + w_src_batch.reshape(K,1) * weight_update
                w_tgt = w_tgt * (1 - weight_update) + w_tgt_batch.reshape(K,1) * weight_update
                w_imp = w_tgt/w_src
                for i in range(w_imp.shape[0]):
                    if w_src[i] == 0:
                        w_imp[i] = 2
                    elif torch.isinf(w_imp[i]):
                        w_imp[i] = 1
                
                    
            #backprop feature extraction+classifier
            feature_source_optim.zero_grad()
            feature_target_optim.zero_grad()
            clf_optim.zero_grad()
            loss.backward()
            feature_source_optim.step()
            feature_target_optim.step()
            clf_optim.step()
    return model_feature_source, model_feature_target, model_clf


def evaluate(
    model_feature_source: nn.Module, \
    model_feature_target: nn.Module, \
    model_clf: nn.Module,\
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
    
    model_feature_source = model_feature_source.eval()
    model_feature_target = model_feature_target.eval()
    model_clf = model_clf.eval()
    total_accuracy = 0
    K = 10
    with torch.no_grad():
        for x, y_true in tqdm(eval_dataloader_all, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            h_t = model_feature_target(x)
            y_pred = model_clf(h_t)
            cluster_t = F.one_hot(torch.argmax(y_pred,1),num_classes=K).float()
            cluster_t = np.argmax(cluster_t.cpu().detach().numpy(), axis=1)
            cluster_df = pd.DataFrame(cluster_t)
            total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()

    return total_accuracy

def train_evaluate(parameterization):
    seed_new = 123
    seed_torch(seed=seed_new)
    
    model_feature_source, model_feature_target, model_clf = train(Net = Net,\
                                                                  Classifier = Classifier,\
                                                                  source_loader=source_loader,\
                                                                  target_loader=target_loader,
                                                                  parameters=parameterization)
    return evaluate(
        model_feature_source=model_feature_source,
        model_feature_target=model_feature_target,
        model_clf = model_clf,\
        eval_dataloader_all = eval_dataloader_all
    )

best_parameters, values, experiment, model = optimize(
    parameters=[
        #{"name": "lr", "type": "range", "bounds": [1e-6, 1e-3]},
        {"name": "lamb_clf", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "lamb_wd", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "lamb_centroid", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "lamb_sntg", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
    ],
    evaluation_function=train_evaluate,
    objective_name='accuracy',
    total_trials=10
)


print("best_parameters",best_parameters)
means, covariances = values
print("means",means)
print("covariances",covariances)

