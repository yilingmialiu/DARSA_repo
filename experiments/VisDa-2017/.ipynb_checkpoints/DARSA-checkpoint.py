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
from torch.nn.utils import spectral_norm
import utils.config as config
from utils.helper import *
from geomloss import SamplesLoss
import pandas as pd
import random
import time
import warnings
import argparse
import shutil
import os.path as osp
import os
from utils.helperOH import *
import data.officehome as datasets
import models.model_resnet as models
import timm
from torch.utils.data import DataLoader
from models.model_office_home import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_new = 123 #666 #123
seed_torch(seed=seed_new)
    
def train(batch_size,lr,K,lamb_clf,lamb_wd,lamb_centroid,lamb_sntg,epochs,LAMBDA,momentum,alpha,source_num,target_num,eval_target_num):
    best_accu_t = 0.0
    # Weight update value
    weight_update = 1
    num_classes = 65
    bottle_dim = 512
    ##initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = get_model('vit_base_patch16_224', pretrain=False) #vit_base_patch16_224
    pool_layer = nn.Identity()
    model_feature_source = Net(backbone, num_classes, bottleneck_dim=bottle_dim,
                        pool_layer=pool_layer, finetune=False).to(device)
    model_feature_target = Net(backbone, num_classes, bottleneck_dim=bottle_dim,
                        pool_layer=pool_layer, finetune=False).to(device)
    model_clf = Classifier(input_dim=bottle_dim).to(device)
    path_to_model = 'trained_models/'
    model_feature_source.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_sntg_0_512.pt'))
    model_feature_target.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_sntg_0_512.pt'))
    model_clf.load_state_dict(torch.load(path_to_model+'source_clf_rd_128_SGD_sntg_0_512.pt'))

    half_batch = batch_size // 2
    ##########source data########################
    train_resizing = 'default'
    val_resizing = 'default'
    
    train_transform = get_train_transform(train_resizing, random_horizontal_flip=False,
                                          random_color_jitter=False, resize_size=224,
                                          norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225))
    val_transform = get_val_transform(val_resizing, resize_size=False,
                                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225))
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names = \
    get_dataset(dataset_name='OfficeHome', root='/datacommons/carlsonlab/yl407/office_home_imbalance',\
                source={'Ar_s': 'image_list/Art_s.txt'}, target={'Cl_t': 'image_list/Clipart_t.txt'}, \
                train_source_transform=train_transform, val_transform=val_transform)
    
    print(len(train_source_dataset))
    print(len(train_target_dataset))
    
    #########source data############################
    source_dataset = train_source_dataset
    source_loader = DataLoader(source_dataset, batch_size=half_batch, drop_last=True,\
                               shuffle=True, num_workers=0, pin_memory=True)
    
    ##########target data########################
    target_dataset_label = train_target_dataset
    target_loader = DataLoader(target_dataset_label, batch_size=half_batch,drop_last=True,\
                               shuffle=True, num_workers=0, pin_memory=True)
    
    ##########evaluation data########################
    eval_dataset = test_dataset

    eval_dataloader_all = DataLoader(eval_dataset, batch_size=32, shuffle=False,
                            drop_last=False, num_workers=0, pin_memory=True)        

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
        print("epoch",epoch)
        source_batch_iterator = ForeverDataIterator(source_loader)
        target_batch_iterator = ForeverDataIterator(target_loader)

        total_unweight_clf_loss = 0
        total_clf_loss = 0
        total_centroid_loss = 0
        total_sntg_loss = 0
        total_w1_loss = 0
        total_w1_original = 0
        
        len_loader = 0
        for i in range(100):
            len_loader = len_loader + 1
            data_source = next(source_batch_iterator)
            source_x, source_y = data_source
            data_target = next(target_batch_iterator)
            target_x, _ = data_target
            source_x, target_x = source_x.to(device), target_x.to(device)
                      
            set_requires_grad(model_feature_source, requires_grad=True)
            set_requires_grad(model_feature_target, requires_grad=True)
            set_requires_grad(model_clf, requires_grad=True)
            
            model_feature_source = model_feature_source.train()
            model_feature_target = model_feature_target.train()
            model_clf = model_clf.train()
            
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

            wasserstein_distance_all = w1_loss(source_feature,target_feature)
            ##clustering loss
            #L_orthogonal
            source_sntg_loss = sntg_loss_func(cluster=cluster_s,feature=source_feature,LAMBDA=LAMBDA)
            target_sntg_loss = sntg_loss_func(cluster=cluster_t,feature=target_feature,LAMBDA=LAMBDA)

            ##calculate centroids
            centroid_loss = centroid_loss_func(K,device,source_y,target_y,source_feature,target_feature)
            sntg_loss = source_sntg_loss + target_sntg_loss
            if torch.isnan(centroid_loss) == True:
                loss = lamb_clf*clf_loss + lamb_wd * wasserstein_distance + lamb_sntg*sntg_loss 
            else:
                loss = lamb_clf*clf_loss + lamb_wd * wasserstein_distance + lamb_centroid*centroid_loss + lamb_sntg*sntg_loss 

            #update weights
            with torch.no_grad():
                w_src_batch = cluster_s.mean(dim=0) 
                w_tgt_batch = cluster_t.mean(dim=0)
                w_src = w_src * (1 - weight_update) + w_src_batch.reshape(K,1) * weight_update
                w_tgt = w_tgt * (1 - weight_update) + w_tgt_batch.reshape(K,1) * weight_update
                w_imp = w_tgt/w_src
                for i in range(w_imp.shape[0]):
                    if w_src[i] == 0 and w_tgt[i] != 0:
                        w_imp[i] = 2
                    elif w_src[i] == 0 and w_tgt[i] == 0:
                        w_imp[i] = 1
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
            

            #total_w1_loss += wasserstein_distance.item()
            total_unweight_clf_loss += report_clf_loss_unweight.item()
            total_clf_loss += clf_loss.item()
            if torch.isnan(centroid_loss) == False:
                total_centroid_loss += centroid_loss.item()
            total_sntg_loss += sntg_loss.item()
            total_w1_original += wasserstein_distance_all.item()
            
            
        mean_clf_loss = total_clf_loss/(len_loader)
        mean_unweighted_clf_loss = total_unweight_clf_loss/(len_loader)
        mean_centroid_loss = total_centroid_loss/(len_loader)
        mean_sntg_loss = total_sntg_loss/(len_loader)
        #mean_w1_loss = total_w1_loss/(len_dataloader)
        mean_w1_original = total_w1_original/(len_loader)

    
        mean_clf_loss_all.append(mean_clf_loss)
        mean_centroid_loss_all.append(mean_centroid_loss)
        mean_sntg_loss_all.append(mean_sntg_loss)
        mean_unweighted_clf_loss_all.append(mean_unweighted_clf_loss)
        #mean_w1_loss_all.append(mean_w1_loss)
        mean_w1_original_all.append(mean_w1_original)
        
        #print(mean_w1_original_all)

        tqdm.write(f'EPOCH {epoch:03d}: critic_loss={mean_w1_original:.4f} source_clf={mean_clf_loss:.4f},unweighted_source_clf={mean_unweighted_clf_loss:.4f},clustering_center={mean_centroid_loss:.4f},sntg={mean_sntg_loss:.4f}')
                        
        ##evaluate models on target domain##
        if epoch %1 == 0: 
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
                    #print(cluster_df.iloc[:,0].value_counts())
                    #if epoch ==1:
                    #    cluster_df_true = pd.DataFrame(y_true.cpu().detach().numpy())
                    #    print(cluster_df_true.iloc[:,0].value_counts())
                    total_accuracy += (y_pred.max(1)[1] == y_true).float().sum().item()
            
            
            mean_accuracy = total_accuracy / len(eval_dataset)
            mean_accuracy_all.append(mean_accuracy)
            print(f'Accuracy on target data: {mean_accuracy:.4f}')
        
        
        if mean_accuracy > best_accu_t:
            best_accu_t = mean_accuracy
        print(best_accu_t)


if __name__ == '__main__':
    K = 65
    batch_size = 16 #1024 #2048 #1024
    lr = 1e-5   #0.000276 #0.0008 #0.0001
    lamb_clf = 1 #0.8737  #1 #0.94 #0.62 #0.9 #0.4 
    lamb_wd = 0.358 #0.45 #0.4 #0.4 
    lamb_centroid = 0 #0 #0.0001 #1 #1
    lamb_sntg = 0.2290  #0.2189 #0 #0.0001 #1
    LAMBDA = 30
    momentum = 0.4208 #0.3479 #0.34 #0.9
    alpha = 8
    source_num = 600 #500
    target_num = 100 #500
    eval_target_num = 10
    print("momentum",momentum)
    print("K",K)
    print("batch_size",batch_size)
    print("lr",lr)
    print("lambda_clf",lamb_clf)
    print("lambda_wd",lamb_wd)
    print("lambda_centroid",lamb_centroid)
    print("lambda_sntg",lamb_sntg)
    print("LAMBDA",LAMBDA)
    print("alpha",alpha)
    train(batch_size=batch_size,lr=lr,\
          K=K,lamb_clf=lamb_clf,lamb_wd=lamb_wd,\
          lamb_centroid=lamb_centroid,lamb_sntg=lamb_sntg,\
          epochs=100,LAMBDA=LAMBDA,momentum=momentum,\
          alpha = alpha,\
          source_num=source_num,target_num=target_num,\
          eval_target_num=eval_target_num)
