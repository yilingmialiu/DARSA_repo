"""
Implements WDGRL with clustering
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""

###library loading###
import sys
sys.path.insert(0, '../../')
sys.path.append('/datacommons/carlsonlab/yl407/packages')
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
from data.svhn import *
from torch.nn.utils import spectral_norm
from models.model_svhn_mnist import *
import utils.config as config
from utils.helper import *
from geomloss import SamplesLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_new = 123
seed_torch(seed=seed_new)

def train(batch_size,lr,K,lamb_clf,lamb_wd,lamb_centroid,lamb_sntg,epochs,LAMBDA,momentum):
    seed_new = 123
    seed_torch(seed=seed_new)
    
    best_accu_t = 0.0
    # Weight update value
    weight_update = 1
    
    ##initialize models
    model_feature_source = Net().to(device)
    model_feature_target = Net().to(device)
    model_clf = Classifier().to(device)
    path_to_model = 'trained_models/'
    model_feature_source.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_sntg.pt'))
    model_feature_target.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_sntg.pt'))
    model_clf.load_state_dict(torch.load(path_to_model+'source_clf_rd_128_SGD_sntg.pt'))

    half_batch = batch_size // 2
    ##########source data########################
    source_dataset = load_SVHN_LS(target_num=1500,train_flag='train')
    source_loader = DataLoader(source_dataset, batch_size=half_batch, drop_last=True,\
                               shuffle=True, num_workers=0, pin_memory=True)
    
    ##########target data########################
    target_dataset_label = load_mnist_LS(source_num=290,train_flag=False)
    
    target_loader = DataLoader(target_dataset_label, batch_size=half_batch,drop_last=True,\
                               shuffle=True, num_workers=0, pin_memory=True)
    
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
    
    ##initialize losses
    mean_w1_loss_all = []
    mean_w1_original_all = []
    mean_clf_loss_all = []
    mean_unweighted_clf_loss_all = []
    mean_centroid_loss_all = []
    mean_sntg_loss_all = []
    mean_accuracy_all = []
            
    for epoch in range(1, epochs+1):
        source_batch_iterator = iter(source_loader)
        target_batch_iterator = iter(target_loader)
        len_dataloader = min(len(source_loader), len(target_loader))
        
        total_unweight_clf_loss = 0
        total_clf_loss = 0
        total_centroid_loss = 0
        total_sntg_loss = 0
        total_w1_loss = 0
        total_w1_original = 0


        for i in range(len_dataloader):
            
            data_source = next(source_batch_iterator)
            source_x, source_y = data_source
            data_target = next(target_batch_iterator)
            target_x, target_ground_truth = data_target
            source_x, target_x = source_x.to(device), target_x.to(device)
                      
            set_requires_grad(model_feature_source, requires_grad=True)
            set_requires_grad(model_feature_target, requires_grad=True)
            set_requires_grad(model_clf, requires_grad=True)
            
            model_feature_source = model_feature_source.train()
            model_feature_target = model_feature_target.train()
            model_clf = model_clf.train()
            
            ##extract latent features
            source_y = source_y.to(torch.int64).to(device)
            target_ground_truth = target_ground_truth.to(torch.int64).to(device)
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
            cluster_s_gt = F.one_hot(source_y, num_classes=K).float()
            target_y = target_preds.to(torch.int64).to(device)
            cluster_t = F.one_hot(target_y, num_classes=K).float()
            cluster_t_gt = F.one_hot(target_ground_truth, num_classes=K).float()
            
            #update weights
            with torch.no_grad():
                w_src_batch = cluster_s_gt.mean(dim=0)
                w_tgt_batch = cluster_t_gt.mean(dim=0)
                if epoch == 1:
                    print("B4",w_tgt_batch)
                w_tgt_batch = w_tgt_batch*1.5
                if epoch == 1:
                    print("After",w_tgt_batch)
                w_src = w_src * (1 - weight_update) + w_src_batch.reshape(K,1) * weight_update
                w_tgt = w_tgt * (1 - weight_update) + w_tgt_batch.reshape(K,1) * weight_update
                w_imp = w_tgt/w_src
            
            ##weighted classification loss
            weighted_clf_err = cluster_s * clf_loss_unweight.reshape(-1,1)
            expected_clf_err = torch.mean(weighted_clf_err, dim=0)
            clf_loss = torch.sum(expected_clf_err.reshape(K,1) * w_imp)

            ##weighted domain invariant loss   
            wasserstein_distance = 0
            for cluster_id in range(K):
                if torch.sum(target_preds==cluster_id)!=0:
                    wasserstein_distance += w_tgt[cluster_id]*w1_loss(source_feature[source_y==cluster_id,],\
                                             target_feature[target_preds==cluster_id,]) 
   
            wasserstein_distance_all = w1_loss(source_feature,target_feature)
            ##clustering loss
            #L_orthogonal
            source_sntg_loss = sntg_loss_func(cluster=cluster_s,feature=source_feature,LAMBDA=LAMBDA)
            target_sntg_loss = sntg_loss_func(cluster=cluster_t,feature=target_feature,LAMBDA=LAMBDA)

            ##calculate centroids
            centroid_loss = centroid_loss_func(K,device,source_y,target_y,source_feature,target_feature)
            sntg_loss = source_sntg_loss + target_sntg_loss


            loss = lamb_clf*clf_loss + lamb_wd * wasserstein_distance + lamb_centroid*centroid_loss + lamb_sntg*sntg_loss

            #backprop feature extraction+classifier
            feature_source_optim.zero_grad()
            feature_target_optim.zero_grad()
            clf_optim.zero_grad()
            loss.backward()
            feature_source_optim.step()
            feature_target_optim.step()
            clf_optim.step()
                
            total_w1_loss += wasserstein_distance.item()
            total_unweight_clf_loss += report_clf_loss_unweight.item()
            total_clf_loss += clf_loss.item()
            total_centroid_loss += centroid_loss.item()
            total_sntg_loss += sntg_loss.item()
            total_w1_original += wasserstein_distance_all.item()
            
            
        mean_clf_loss = total_clf_loss/(len_dataloader)
        mean_unweighted_clf_loss = total_unweight_clf_loss/(len_dataloader)
        mean_centroid_loss = total_centroid_loss/(len_dataloader)
        mean_sntg_loss = total_sntg_loss/(len_dataloader)
        mean_w1_loss = total_w1_loss/(len_dataloader)
        mean_w1_original = total_w1_original/(len_dataloader)

    
        mean_clf_loss_all.append(mean_clf_loss)
        mean_centroid_loss_all.append(mean_centroid_loss)
        mean_sntg_loss_all.append(mean_sntg_loss)
        mean_unweighted_clf_loss_all.append(mean_unweighted_clf_loss)
        mean_w1_loss_all.append(mean_w1_loss)
        mean_w1_original_all.append(mean_w1_original)

    
        tqdm.write(f'EPOCH {epoch:03d}: critic_loss={mean_w1_loss:.4f} source_clf={mean_clf_loss:.4f},unweighted_source_clf={mean_unweighted_clf_loss:.4f},clustering_centr={mean_centroid_loss:.4f},sntg={mean_sntg_loss:.4f}')
                
            
        eval_dataset = load_mnist_LS(source_num=290,train_flag=False)

        eval_dataloader_all = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False,
                            drop_last=False, num_workers=0, pin_memory=True)
        
        ##evaluate models on target domain##
        set_requires_grad(model_feature_source, requires_grad=False)
        set_requires_grad(model_feature_target, requires_grad=False)
        set_requires_grad(model_clf, requires_grad=False)
        
        model_feature_source = model_feature_source.eval()
        model_feature_target = model_feature_target.eval()
        model_clf = model_clf.eval()
        
        
        total_accuracy = 0
        with torch.no_grad():
            for x, y_true in tqdm(eval_dataloader_all, leave=False):
                x, y_true = x.to(device), y_true.to(device)
                h_t = model_feature_target(x)
                y_pred = model_clf(h_t)
                cluster_t = F.one_hot(torch.argmax(y_pred,1),num_classes=K).float()
                cluster_t = np.argmax(cluster_t.cpu().detach().numpy(), axis=1)
                cluster_df = pd.DataFrame(cluster_t)
                print(cluster_df.iloc[:,0].value_counts())
                if epoch ==1:
                    cluster_df_true = pd.DataFrame(y_true.cpu().detach().numpy())
                    print(cluster_df_true.iloc[:,0].value_counts())
                total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
            
            
        mean_accuracy = total_accuracy / len(eval_dataloader_all)
        mean_accuracy_all.append(mean_accuracy)
        print(f'Accuracy on target data: {mean_accuracy:.4f}')
                
        
        if mean_accuracy > best_accu_t:
            best_accu_t = mean_accuracy
            
        print(best_accu_t)


if __name__ == '__main__':
    K = 10
    batch_size = 512
    lr = 0.01
    lamb_clf = 0.95
    lamb_wd = 0.11
    lamb_centroid = 0.11
    lamb_sntg = 0.3
    LAMBDA = 50
    momentum = 0.4
    print("momentum",momentum)
    print("K",K)
    print("batch_size",batch_size)
    print("lr",lr)
    print("lambda_clf",lamb_clf)
    print("lambda_wd",lamb_wd)
    print("lambda_centroid",lamb_centroid)
    print("lambda_sntg",lamb_sntg)
    print("LAMBDA",LAMBDA)
    train(batch_size=batch_size,lr=lr,\
          K=K,lamb_clf=lamb_clf,lamb_wd=lamb_wd,\
          lamb_centroid=lamb_centroid,lamb_sntg=lamb_sntg,\
          epochs=1000,LAMBDA=LAMBDA,momentum=momentum)
