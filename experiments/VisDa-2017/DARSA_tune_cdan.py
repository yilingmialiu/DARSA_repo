"""
Implements WDGRL with clustering
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""

###library loading###
import sys
import os

sys.path.insert(0, '../../')
sys.path.append('/datacommons/carlsonlab/yl407/packages')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, project_root)


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
from geomloss import SamplesLoss
import pandas as pd
import random
import time
import warnings
import argparse
import shutil
import os.path as osp
import os
import timm
from torch.utils.data import DataLoader
from utils.helper import *
from utils.helperVisDa import *
from data.visda2017 import VisDA2017
from models.model_visda import *
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_new = 123 #666 #123
seed_torch(seed=seed_new)
    
def train(batch_size, lr, K, lamb_clf, lamb_wd, lamb_centroid, lamb_sntg, epochs, LAMBDA, momentum, alpha, source_num, target_num, eval_target_num, lr_gamma=0.001, lr_decay=0.75):
    best_accu_t = 0.0
    weight_update = 1
    num_classes = 12  # VisDA-2017 has 12 classes
    bottle_dim = 256  # Changed to match source training

    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create source model
    backbone_source = get_model('vit_base_patch16_224', pretrain=True)
    full_model_source = ImageClassifier(backbone_source, num_classes, bottleneck_dim=bottle_dim,
                                      pool_layer=nn.Identity(), finetune=False).to(device)
    
    # Split source model into feature extractor and classifier
    model_feature_source = nn.Sequential(
        full_model_source.backbone,
        full_model_source.pool_layer,
        full_model_source.bottleneck
    ).to(device)
    
    # Create target model
    backbone_target = get_model('vit_base_patch16_224', pretrain=True)
    full_model_target = ImageClassifier(backbone_target, num_classes, bottleneck_dim=bottle_dim,
                                      pool_layer=nn.Identity(), finetune=False).to(device)
    
    # Split target model into feature extractor and classifier
    model_feature_target = nn.Sequential(
        full_model_target.backbone,
        full_model_target.pool_layer,
        full_model_target.bottleneck
    ).to(device)
    
    model_clf = full_model_source.head.to(device)

    # Create temporary full model to load the checkpoint
    backbone_temp = get_model('vit_base_patch16_224', pretrain=True)
    temp_model = ImageClassifier(backbone_temp, num_classes, bottleneck_dim=bottle_dim,
                               pool_layer=nn.Identity(), finetune=False).to(device)
    
    # Load the checkpoint
    checkpoint_path = '/home/yilingliu/ELS/Domain_Adaptation/logs/cdan_vit/VisDA2017/checkpoints/best.pth'
    print(f"Loading pre-trained model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    print("Checkpoint keys:")
    for k in checkpoint.keys():
        print(f"  {k}")

    # First load the checkpoint into the temporary model
    temp_model.load_state_dict(checkpoint)

    # Now transfer the weights correctly
    # For source feature extractor
    model_feature_source[0].load_state_dict(temp_model.backbone.state_dict())  # backbone
    model_feature_source[1].load_state_dict(temp_model.pool_layer.state_dict())  # pool_layer
    model_feature_source[2].load_state_dict(temp_model.bottleneck.state_dict())  # bottleneck

    # For target feature extractor
    model_feature_target[0].load_state_dict(temp_model.backbone.state_dict())  # backbone
    model_feature_target[1].load_state_dict(temp_model.pool_layer.state_dict())  # pool_layer
    model_feature_target[2].load_state_dict(temp_model.bottleneck.state_dict())  # bottleneck

    # For classifier
    model_clf.load_state_dict(temp_model.head.state_dict())

    print("Successfully loaded and split pre-trained model")

    # Clean up
    del temp_model
    del backbone_temp
    torch.cuda.empty_cache()

    half_batch = batch_size // 2

    # Data loading for VisDA-2017
    train_resizing = 'default'
    val_resizing = 'default'

    train_transform = get_train_transform(train_resizing, 
                                        random_horizontal_flip=False,
                                        random_color_jitter=False, 
                                        resize_size=224,
                                        norm_mean=(0.485, 0.456, 0.406), 
                                        norm_std=(0.229, 0.224, 0.225))
    
    val_transform = get_val_transform(val_resizing, 
                                    resize_size=224,
                                    norm_mean=(0.485, 0.456, 0.406), 
                                    norm_std=(0.229, 0.224, 0.225))
                                    

    # Update dataset paths for VisDA-2017
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names = \
    get_dataset(dataset_name='VisDA2017', 
                root='/data/home/yilingliu/VisDA-2017',
                source={'Synthetic': 'image_list/train_list_imbalanced.txt'}, 
                target={'Real': 'image_list/validation_list_imbalanced.txt'}, 
                train_source_transform=train_transform, 
                val_transform=val_transform)

    print("Source dataset size:", len(train_source_dataset))
    print("Target dataset size:", len(train_target_dataset))
    print("Val dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    # Create data loaders
    source_loader = DataLoader(train_source_dataset, 
                             batch_size=half_batch, 
                             drop_last=True,
                             shuffle=True, 
                             num_workers=2, 
                             pin_memory=True)
    
    target_loader = DataLoader(train_target_dataset, 
                             batch_size=half_batch,
                             drop_last=True,
                             shuffle=True, 
                             num_workers=2, 
                             pin_memory=True)
    
    eval_dataset = test_dataset
    eval_dataloader_all = DataLoader(test_dataset, 
                                   batch_size=32, 
                                   shuffle=False,
                                   drop_last=False, 
                                   num_workers=2, 
                                   pin_memory=True)

    ##initialize model optimizers
    feature_source_optim = torch.optim.SGD(model_feature_source.parameters(), lr=lr, momentum=momentum) 
    feature_target_optim = torch.optim.SGD(model_feature_target.parameters(), lr=lr, momentum=momentum) 
    clf_optim = torch.optim.SGD(model_clf.parameters(), lr=lr, momentum=momentum)

    # Initialize learning rate schedulers
    feature_source_scheduler = LambdaLR(feature_source_optim,
                                      lambda x: lr * (1. + lr_gamma * float(x)) ** (-lr_decay))
    feature_target_scheduler = LambdaLR(feature_target_optim,
                                      lambda x: lr * (1. + lr_gamma * float(x)) ** (-lr_decay))
    clf_scheduler = LambdaLR(clf_optim,
                              lambda x: lr * (1. + lr_gamma * float(x)) ** (-lr_decay))
    
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
            
    # After loading the pre-trained model and before starting training
    print("\nEvaluating initial model performance on target domain...")
    
    model_feature_target.eval()
    model_clf.eval()
    
    # Initialize per-class accuracy tracking
    per_class_correct = torch.zeros(K).to(device)
    per_class_total = torch.zeros(K).to(device)
    total_accuracy = 0
    
    with torch.no_grad():
        for x, y_true in tqdm(eval_dataloader_all, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            h_t = model_feature_target(x)
            y_pred = model_clf(h_t)
            
            # Update total accuracy
            total_accuracy += (y_pred.max(1)[1] == y_true).float().sum().item()
            
            # Update per-class accuracy
            predictions = y_pred.max(1)[1]
            for label in range(K):
                per_class_correct[label] += ((predictions == label) & (y_true == label)).float().sum()
                per_class_total[label] += (y_true == label).float().sum()
    
    # Calculate and print results
    mean_accuracy = total_accuracy / len(eval_dataset)
    per_class_accuracies = per_class_correct / per_class_total
    mean_class_accuracy = per_class_accuracies.mean().item()
    
    print(f'Initial Model Performance:')
    print(f'Average Accuracy = {mean_accuracy:.4f}')
    print(f'Average Class Accuracy = {mean_class_accuracy:.4f}')
    print('Per-Class Accuracies:')
    for i in range(K):
        print(f'    {class_names[i]}: {per_class_accuracies[i].item():.4f}')
    print('-' * 50)

    for epoch in range(1, epochs+1):
        print("epoch", epoch)
        source_batch_iterator = ForeverDataIterator(source_loader)
        target_batch_iterator = ForeverDataIterator(target_loader)

        total_unweight_clf_loss = 0
        total_clf_loss = 0
        total_centroid_loss = 0
        total_sntg_loss = 0
        total_w1_loss = 0
        total_w1_original = 0
        
        # Calculate number of iterations needed to go through the entire dataset
        num_iterations = max(len(source_loader), len(target_loader))
        
        for i in tqdm(range(num_iterations)):
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
            
            
        mean_clf_loss = total_clf_loss / num_iterations
        mean_unweighted_clf_loss = total_unweight_clf_loss / num_iterations
        mean_centroid_loss = total_centroid_loss / num_iterations
        mean_sntg_loss = total_sntg_loss / num_iterations
        mean_w1_original = total_w1_original / num_iterations

    
        mean_clf_loss_all.append(mean_clf_loss)
        mean_centroid_loss_all.append(mean_centroid_loss)
        mean_sntg_loss_all.append(mean_sntg_loss)
        mean_unweighted_clf_loss_all.append(mean_unweighted_clf_loss)
        mean_w1_original_all.append(mean_w1_original)
        

        tqdm.write(f'EPOCH {epoch:03d}: critic_loss={mean_w1_original:.4f} source_clf={mean_clf_loss:.4f},unweighted_source_clf={mean_unweighted_clf_loss:.4f},clustering_center={mean_centroid_loss:.4f},sntg={mean_sntg_loss:.4f}')
                        
        ##evaluate models on target domain##
        if epoch % 1 == 0: 
            set_requires_grad(model_feature_source, requires_grad=False)
            set_requires_grad(model_feature_target, requires_grad=False)
            set_requires_grad(model_clf, requires_grad=False)
        
            model_feature_source = model_feature_source.eval()
            model_feature_target = model_feature_target.eval()
            model_clf = model_clf.eval()
        
            total_accuracy = 0
            # Initialize per-class accuracy tracking
            per_class_correct = torch.zeros(K).to(device)
            per_class_total = torch.zeros(K).to(device)
            
            with torch.no_grad():
                for x, y_true in tqdm(eval_dataloader_all, leave=False):
                    x, y_true = x.to(device), y_true.to(device)
                    h_t = model_feature_target(x)
                    y_pred = model_clf(h_t)
                    cluster_t = F.one_hot(torch.argmax(y_pred,1), num_classes=K).float()
                    cluster_t = np.argmax(cluster_t.cpu().detach().numpy(), axis=1)
                    cluster_df = pd.DataFrame(cluster_t)
                    
                    # Update total accuracy
                    total_accuracy += (y_pred.max(1)[1] == y_true).float().sum().item()
                    
                    # Update per-class accuracy
                    predictions = y_pred.max(1)[1]
                    for label in range(K):
                        per_class_correct[label] += ((predictions == label) & (y_true == label)).float().sum()
                        per_class_total[label] += (y_true == label).float().sum()
            
            # Calculate mean accuracy and per-class accuracies
            mean_accuracy = total_accuracy / len(eval_dataset)
            per_class_accuracies = per_class_correct / per_class_total
            mean_class_accuracy = per_class_accuracies.mean().item()
            
            # Print results
            print(f'Epoch: {epoch:03d}')
            print(f'Average Accuracy = {mean_accuracy:.4f}')
            print(f'Average Class Accuracy = {mean_class_accuracy:.4f}')
            print('Class Accuracies:')
            for i in range(K):
                print(f'    {class_names[i]}: {per_class_accuracies[i].item():.4f}')
            
            mean_accuracy_all.append(mean_accuracy)
            
            if mean_accuracy > best_accu_t:
                best_accu_t = mean_accuracy
                # Save best model
                print('Saving best model...')
                torch.save({
                    'epoch': epoch,
                    'feature_target_state_dict': model_feature_target.state_dict(),
                    'classifier_state_dict': model_clf.state_dict(),
                    'best_accuracy': best_accu_t,
                    'per_class_accuracies': per_class_accuracies.cpu().numpy()
                }, f'trained_models/visda_target_model_best_sntg.pth')
                
            print(f'Best accuracy so far: {best_accu_t:.4f}')

        # After training loop in each epoch, step the schedulers
        feature_source_scheduler.step()
        feature_target_scheduler.step()
        clf_scheduler.step()
        
        # Print current learning rates
        current_lr_source = feature_source_optim.param_groups[0]['lr']
        current_lr_target = feature_target_optim.param_groups[0]['lr']
        current_lr_clf = clf_optim.param_groups[0]['lr']
        print(f'Learning rates: source={current_lr_source:.6f}, target={current_lr_target:.6f}, clf={current_lr_clf:.6f}')


if __name__ == '__main__':
    K = 12  # Number of classes in VisDA-2017
    batch_size = 64  # Adjusted for VisDA-2017
    lr = 0.001
    lamb_clf = 1
    lamb_wd = 0.1
    lamb_centroid = 0 #0.01
    lamb_sntg = 0 #0.01
    LAMBDA = 30
    momentum = 0.9
    alpha = 8
    source_num = 152397  # Total number of synthetic images
    target_num = 55388   # Total number of real images
    eval_target_num = 5000  # Number of validation samples
    lr_gamma = 0.001  # Added learning rate gamma parameter
    lr_decay = 0.75   # Added learning rate decay parameter

    print("momentum", momentum)
    print("K", K)
    print("batch_size", batch_size)
    print("lr", lr)
    print("lr_gamma", lr_gamma)
    print("lr_decay", lr_decay)
    print("lambda_clf", lamb_clf)
    print("lambda_wd", lamb_wd)
    print("lambda_centroid", lamb_centroid)
    print("lambda_sntg", lamb_sntg)
    print("LAMBDA", LAMBDA)
    print("alpha", alpha)
    train(batch_size=batch_size, lr=lr,
          K=K, lamb_clf=lamb_clf, lamb_wd=lamb_wd,
          lamb_centroid=lamb_centroid, lamb_sntg=lamb_sntg,
          epochs=100, LAMBDA=LAMBDA, momentum=momentum,
          alpha=alpha,
          source_num=source_num, target_num=target_num,
          eval_target_num=eval_target_num,
          lr_gamma=lr_gamma, lr_decay=lr_decay)
