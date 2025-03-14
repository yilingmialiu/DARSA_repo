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
import torch
import numpy as np
from torch import nn
import math
import pandas as pd
import os
import sys
import pickle as pkl
sys.path.insert(0, '../../')
import itertools
from torch.autograd import grad
from torch.utils.data import DataLoader,Subset
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torchvision import datasets, transforms
from models.model_neural import *
import utils.config as config
from utils.helper import *
from geomloss import SamplesLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from typing import Dict, List, Optional, Tuple

seed_new = 123
seed_torch(seed=seed_new)

batch_size = 512
half_batch = batch_size // 2

dataDir = "/datacommons/carlsonlab/yl407/TST_data/"
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
num_b = [6000,3000,6000] #[3000,6000,8000]
indices_source_0 = np.random.choice(np.where((y_source == 0))[0],num_b[0],replace=False)
indices_source_1 = np.random.choice(np.where((y_source == 1))[0],num_b[1],replace=False)
indices_source_2 = np.random.choice(np.where((y_source == 2))[0],num_b[2],replace=False)
indices_source = np.concatenate((indices_source_0,indices_source_1,indices_source_2))
np.random.shuffle(indices_source)
X_source = X_source[indices_source]
y_source = y_source[indices_source]

num_w = [3000,6000,3000] #[6000,3000,16000]
indices_target_0 = np.random.choice(np.where((y_target == 0))[0],num_w[0],replace=False)
indices_target_1 = np.random.choice(np.where((y_target == 1))[0],num_w[1],replace=False)
indices_target_2 = np.random.choice(np.where((y_target == 2))[0],num_w[2],replace=False)
indices_target = np.concatenate((indices_target_0,indices_target_1,indices_target_2))
np.random.shuffle(indices_target)
X_target = X_target[indices_target]
y_target = y_target[indices_target]

class my_dataset(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label         
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.data)

##########source data########################
source_dataset = my_dataset(X_source, y_source)
source_loader = DataLoader(source_dataset, batch_size=half_batch, drop_last=True,\
                           shuffle=True, num_workers=0, pin_memory=True)

##########target data########################
target_dataset_label = my_dataset(X_target, y_target)
target_loader = DataLoader(target_dataset_label, batch_size=half_batch,drop_last=True,\
                           shuffle=True, num_workers=0, pin_memory=True)


eval_dataset = my_dataset(X_target, y_target)

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
    
    K = parameters.get("K", 3)
    lr = parameters.get("lr", 1e-4) 
    lamb_clf = parameters.get("lamb_clf", 1) 
    lamb_wd = parameters.get("lamb_wd", 0.5) 
    lamb_centroid = parameters.get("lamb_centroid", 1) 
    lamb_sntg = parameters.get("lamb_sntg", 1)
    LAMBDA = parameters.get("LAMBDA", 50) 
    momentum = parameters.get("momentum", 0.4)  
    epochs= parameters.get("epochs", 20)  
    
    ##initialize models
    model_feature_source = Net(dim_in=616).to(device)
    model_feature_target = Net(dim_in=616).to(device)
    model_clf = Classifier(num_classes=3).to(device)
    path_to_model = 'trained_models/'
    model_feature_source.load_state_dict(torch.load(path_to_model+'source_feature_b_to_w.pt'))
    model_feature_target.load_state_dict(torch.load(path_to_model+'source_feature_b_to_w.pt'))
    model_clf.load_state_dict(torch.load(path_to_model+'source_clf_b_to_w.pt'))
    
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
            
            data_source = source_batch_iterator.next()
            source_x, source_y = data_source
            data_target = target_batch_iterator.next()
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
    
    model_feature_source, model_feature_target, model_clf = train(Net = DenseNet,\
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
        {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2]},
        {"name": "lamb_clf", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "lamb_wd", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "lamb_centroid", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "lamb_sntg", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
    ],
    evaluation_function=train_evaluate,
    objective_name='accuracy',
)


print("best_parameters",best_parameters)
means, covariances = values
print("means",means)
print("covariances",covariances)

