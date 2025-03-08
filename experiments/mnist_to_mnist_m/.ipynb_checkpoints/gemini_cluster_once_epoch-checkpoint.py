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
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, RandomRotation, RandomAffine
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
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_new = 1234
seed_torch(seed=seed_new)

def train(batch_size, lr, K, lamb_clf, lamb_wd, lamb_centroid, lamb_sntg, epochs, LAMBDA, momentum, alpha, source_num, target_num, eval_target_num, pseudo_label_threshold):

    best_accu_t = 0.0
    # Weight update value
    weight_update = 1

    ##initialize models
    model_feature_source = Net().to(device)
    model_feature_target = Net().to(device)
    model_clf = Classifier().to(device)
    path_to_model = 'trained_models/'

    model_feature_source.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_alpha_'+str(alpha)+'_v4.pt'))
    model_feature_target.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_alpha_'+str(alpha)+'_v4.pt'))
    model_clf.load_state_dict(torch.load(path_to_model+'source_clf_rd_128_SGD_alpha_'+str(alpha)+'_v4.pt'))

    half_batch = batch_size // 2
    ##########source data########################
    source_dataset = load_mnist_LS(source_num=source_num,alpha=alpha,train_flag=True)
    source_loader = DataLoader(source_dataset, batch_size=half_batch, drop_last=True, shuffle=True, num_workers=0, pin_memory=True)

    ##########target data########################
    target_dataset_label = load_mnistm_LS(target_num=target_num,alpha=alpha,MNISTM=MNISTM,train_flag=True)
    target_loader = DataLoader(target_dataset_label, batch_size=half_batch, drop_last=True, shuffle=True, num_workers=0, pin_memory=True)

    ##########evaluation data########################
    eval_dataset = load_mnistm_LS(target_num=eval_target_num,alpha=alpha,MNISTM=MNISTM,train_flag=False)

    eval_dataloader_all = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=True,
                            drop_last=True, num_workers=0, pin_memory=True)

    # Initial evaluation (optional)
    model_feature_source = model_feature_source.eval()
    model_feature_target = model_feature_target.eval()
    model_clf = model_clf.eval()

    total_accuracy = 0
    with torch.no_grad():
        for x, y_true in tqdm(eval_dataloader_all, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            h_t = model_feature_target(x)
            y_pred = model_clf(h_t)
            total_accuracy += (y_pred.max(1)[1] == y_true).float().sum().item()

    mean_accuracy = total_accuracy / len(eval_dataset)
    print(f'Initial accuracy on target eval data: {mean_accuracy:.4f}')

    ##initialize model optimizers
    feature_source_optim = torch.optim.SGD(model_feature_source.parameters(), lr=lr,momentum=momentum)
    feature_target_optim = torch.optim.SGD(model_feature_target.parameters(), lr=lr,momentum=momentum)
    clf_optim = torch.optim.SGD(model_clf.parameters(), lr=lr,momentum=momentum)

    scheduler_source = StepLR(feature_source_optim, step_size=50, gamma=0.1, verbose=True)
    scheduler_target = StepLR(feature_target_optim, step_size=50, gamma=0.1, verbose=True)
    scheduler_clf = StepLR(clf_optim, step_size=50, gamma=0.1, verbose=True)

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
        print("alpha",alpha)
        source_batch_iterator = iter(source_loader)
        target_batch_iterator = iter(target_loader)
        len_dataloader = min(len(source_loader), len(target_loader))

        total_unweight_clf_loss = 0
        total_clf_loss = 0

        all_source_features = []
        all_source_labels = []
        all_target_features = []
        all_target_preds = []

        for i in range(len_dataloader):

            data_source = next(source_batch_iterator)
            source_x, source_y = data_source
            data_target = next(target_batch_iterator)
            target_x, _ = data_target
            source_x, target_x = source_x.to(device), target_x.to(device)

            model_feature_source = model_feature_source.train()
            model_feature_target = model_feature_target.train()
            model_clf = model_clf.train()

            ##extract latent features
            source_y_batch = source_y.to(torch.int64).to(device)
            source_feature_batch = model_feature_source(source_x)
            target_feature_batch = model_feature_target(target_x)
            target_feature_batch_2 = model_feature_target(target_x)

            ##unweighted classification loss
            source_preds_batch = model_clf(source_feature_batch)
            clf_loss_unweight_batch = clf_criterion(source_preds_batch, source_y_batch)
            report_clf_loss_unweight_batch = clf_criterion_unweight(source_preds_batch, source_y_batch)

            # --- Pseudo-label thresholding ---
            target_preds_probs_batch = soft_f(model_clf(target_feature_batch_2))
            max_probs_batch, target_preds_raw_batch = torch.max(target_preds_probs_batch, dim=1)
            confident_mask_batch = max_probs_batch > pseudo_label_threshold

            # Collect data for epoch-level calculations
            all_source_features.append(source_feature_batch)
            all_source_labels.append(source_y_batch)
            all_target_features.append(target_feature_batch)
            all_target_preds.append(target_preds_raw_batch[confident_mask_batch])

            total_unweight_clf_loss += report_clf_loss_unweight_batch.item()
            total_clf_loss += clf_loss_unweight_batch.mean().item() # Accumulate mean for epoch level

            #print(f"Epoch {epoch}, Batch {i}:")
            #print("  Source Feature Shape:", source_feature_batch.shape)
            #print("  Target Feature Shape:", target_feature_batch.shape)
            #print("  Number of Confident Pseudo-labels:", confident_mask_batch.sum())

        # --- Calculate all losses once per epoch ---
        all_source_features_tensor = torch.cat(all_source_features, dim=0).detach() # Detach for stability
        all_source_labels_tensor = torch.cat(all_source_labels, dim=0).detach()
        all_target_features_tensor = torch.cat(all_target_features, dim=0).detach()
        all_target_preds_tensor = torch.cat(all_target_preds, dim = 0).detach()

        print(f"Epoch {epoch}:")
        print("  All Source Features Shape:", all_source_features_tensor.shape)
        print("  All Target Features Shape:", all_target_features_tensor.shape)
        print("  All Target Pseudo-labels Shape:", all_target_preds_tensor.shape)

        ##get clustering information
        cluster_s = F.one_hot(all_source_labels_tensor, num_classes=K).float().to(device)

        if len(all_target_preds_tensor) > 0:
            cluster_t = F.one_hot(all_target_preds_tensor, num_classes=K).float().to(device)
        else:
            cluster_t = torch.zeros((0, K)).to(device)

        ##weighted domain invariant loss
        wasserstein_distance = 0
        if len(all_target_preds_tensor) > 0:
            for cluster_id in range(K):
                source_mask = all_source_labels_tensor == cluster_id
                target_mask = all_target_preds_tensor == cluster_id
                if torch.sum(target_mask) > 0 and torch.sum(source_mask) > 0:
                    wasserstein_distance += w_tgt[cluster_id] * w1_loss(
                        all_source_features_tensor[source_mask,],
                        all_target_features_tensor[target_mask,]
                    )

        wasserstein_distance_all = w1_loss(all_source_features_tensor, all_target_features_tensor)

        ##clustering loss
        print("Shape of cluster_t:", cluster_t.shape)
        print("Unique pseudo-labels:", torch.unique(all_target_preds_tensor))
        print("Number of confident pseudo-labels:", len(all_target_preds_tensor))

        source_sntg_loss = sntg_loss_func(cluster=cluster_s, feature=all_source_features_tensor, LAMBDA=LAMBDA)
        target_sntg_loss = torch.tensor(0.0).to(device)
        if len(cluster_t) > 0:
            target_sntg_loss = sntg_loss_func(cluster=cluster_t, feature=all_target_features_tensor[all_target_preds_tensor.bool()], LAMBDA=LAMBDA)

        ##calculate centroids
        centroid_loss = torch.tensor(0.0).to(device)
        if len(all_target_preds_tensor) > 0:
            centroid_loss = centroid_loss_func(K, device, all_source_labels_tensor, all_target_preds_tensor, all_source_features_tensor, all_target_features_tensor[all_target_preds_tensor.bool()])

        sntg_loss = source_sntg_loss + target_sntg_loss

        # Calculate weighted classification loss at the epoch level
        cluster_s_epoch = F.one_hot(all_source_labels_tensor, num_classes=K).float().to(device)
        weighted_clf_err_epoch = cluster_s_epoch * clf_loss_unweight(model_clf(all_source_features_tensor), all_source_labels_tensor).reshape(-1, 1)
        expected_clf_err_epoch = torch.mean(weighted_clf_err_epoch, dim=0)
        clf_loss_epoch = torch.sum(expected_clf_err_epoch.reshape(K, 1) * w_imp)

        loss_wdgrl_total = lamb_clf * clf_loss_epoch + lamb_wd * wasserstein_distance + lamb_centroid*centroid_loss + lamb_sntg*sntg_loss

        print("  Wasserstein Distance:", wasserstein_distance.item() if isinstance(wasserstein_distance, torch.Tensor) else 0)
        print("  Centroid Loss:", centroid_loss.item() if isinstance(centroid_loss, torch.Tensor) else 0)
        print("  SNTG Loss:", sntg_loss.item())
        print("  Weighted Classification Loss (Epoch):", clf_loss_epoch.item())
        print("  Total WDGRL Loss:", loss_wdgrl_total.item())

        # Update weights (once per epoch)
        with torch.no_grad():
            w_src_epoch = cluster_s.mean(dim=0)
            if len(cluster_t) > 0:
                w_tgt_epoch = cluster_t.mean(dim=0)
            else:
                w_tgt_epoch = torch.zeros(K).to(device)
            w_src = w_src * (1 - weight_update) + w_src_epoch.reshape(K, 1) * weight_update
            w_tgt = w_tgt * (1 - weight_update) + w_tgt_epoch.reshape(K, 1) * weight_update
            w_imp = w_tgt / w_src
            for k_idx in range(w_imp.shape[0]):
                if w_src[k_idx] == 0:
                    w_imp[k_idx] = 8
                elif torch.isinf(w_imp[k_idx]):
                    w_imp[k_idx] = 1

        print("  w_src:", w_src.flatten().tolist())
        print("  w_tgt:", w_tgt.flatten().tolist())
        print("  w_imp:", w_imp.flatten().tolist())

        # Backpropagate the combined loss
        feature_source_optim.zero_grad()
        feature_target_optim.zero_grad()
        clf_optim.zero_grad()
        loss_wdgrl_total.backward()
        feature_source_optim.step()
        feature_target_optim.step()
        clf_optim.step()

        mean_clf_loss = total_clf_loss/(len_dataloader)
        mean_unweighted_clf_loss = total_unweight_clf_loss/(len_dataloader)
        mean_centroid_loss = centroid_loss.item() if isinstance(centroid_loss, torch.Tensor) else 0
        mean_sntg_loss = sntg_loss.item()
        mean_w1_loss = wasserstein_distance.item() if isinstance(wasserstein_distance, torch.Tensor) else 0
        mean_w1_original = wasserstein_distance_all.item()

        mean_clf_loss_all.append(mean_clf_loss)
        mean_centroid_loss_all.append(mean_centroid_loss)
        mean_sntg_loss_all.append(mean_sntg_loss)
        mean_unweighted_clf_loss_all.append(mean_unweighted_clf_loss)
        mean_w1_loss_all.append(mean_w1_loss)
        mean_w1_original_all.append(mean_w1_original)

        tqdm.write(f'EPOCH {epoch:03d}: critic_loss={mean_w1_loss:.4f} source_clf={mean_clf_loss:.4f},unweighted_source_clf={mean_unweighted_clf_loss:.4f},clustering_centr={mean_centroid_loss:.4f},sntg={mean_sntg_loss:.4f}')

        ##evaluate models on target domain (Evaluation Set)##
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
                total_accuracy += (y_pred.max(1)[1] == y_true).float().sum().item()

        mean_accuracy = total_accuracy / len(eval_dataset)
        mean_accuracy_all.append(mean_accuracy)
        print(f'Accuracy on target eval data: {mean_accuracy:.4f}')

        scheduler_source.step()
        scheduler_target.step()
        scheduler_clf.step()

        if mean_accuracy > best_accu_t:
            best_accu_t = mean_accuracy

        print("val best:",best_accu_t)

if __name__ == '__main__':
    # --- Set specific hyperparameters ---
    K = 10
    batch_size = 256 # Reduced batch size
    lr = 0.01  # Example: Set a specific learning rate
    lamb_clf = 0.8 # Example: Set a specific lambda_clf
    lamb_wd = 0.4  # Example: Set a specific lambda_wd
    lamb_centroid = 0.9
    lamb_sntg = 0.9
    LAMBDA = 30
    momentum = 0.5
    alpha = 8
    source_num = 600
    target_num = 100
    eval_target_num = 10
    pseudo_label_threshold = 0.1 #0.1  # --- Set the pseudo-label threshold ---

    print("--- Starting Run with Specific Hyperparameters and Thresholding ---")
    print("lr", lr)
    print("momentum", momentum)
    print("K", K)
    print("batch_size", batch_size)
    print("lambda_clf", lamb_clf)
    print("lambda_wd", lamb_wd)
    print("lambda_centroid", lamb_centroid)
    print("lambda_sntg", lamb_sntg)
    print("LAMBDA", LAMBDA)
    print("alpha", alpha)
    print("pseudo_label_threshold", pseudo_label_threshold)

    train(batch_size=batch_size, lr=lr,
          K=K, lamb_clf=lamb_clf, lamb_wd=lamb_wd,
          lamb_centroid=lamb_centroid, lamb_sntg=lamb_sntg,
          epochs=3000,
          LAMBDA=LAMBDA, momentum=momentum,
          alpha=alpha,
          source_num=source_num, target_num=target_num,
          eval_target_num=eval_target_num,
          pseudo_label_threshold=pseudo_label_threshold) # Pass the threshold to the train function
    print("--- Finished Run ---")