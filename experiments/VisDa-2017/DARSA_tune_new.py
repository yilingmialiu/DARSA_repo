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
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.cuda.empty_cache()  # Clear any existing cache

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

     # Load the best source model trained from DARSA_train_source.py
    path_to_model = 'trained_models/'
    checkpoint = torch.load(path_to_model + 'visda_source_model_best_sntg_0.1.pth')
    
    # Load the state dictionaries from the checkpoint
    model_feature_source.load_state_dict(checkpoint['feature_state_dict'])
    model_feature_target.load_state_dict(checkpoint['feature_state_dict'])  # Initialize target with source weights
    model_clf.load_state_dict(checkpoint['classifier_state_dict'])
    
    print(f"Loaded source model from epoch {checkpoint['epoch']} with best accuracy: {checkpoint['best_accuracy']:.4f}")
    print("Per-class accuracies on source model:")
    for i, acc in enumerate(checkpoint['per_class_accuracies']):
        print(f"Class {i}: {acc:.4f}")

    print("Successfully loaded and split pre-trained model")


    torch.cuda.empty_cache()

    # Modify the data loaders to use smaller batch sizes initially
    initial_batch = batch_size // 2  # Start with 256 for batch_size=512
    half_batch = batch_size         # Target the full batch size (512)
    warmup_iterations = 10          # Shorter warmup since we need larger batches
    
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

    # Create data loaders with smaller initial batch sizes
    source_loader = DataLoader(train_source_dataset, 
                             batch_size=initial_batch, 
                             drop_last=True,
                             shuffle=True, 
                             num_workers=2, 
                             pin_memory=False)  # Disable pin_memory
    
    target_loader = DataLoader(train_target_dataset, 
                             batch_size=initial_batch,
                             drop_last=True,
                             shuffle=True, 
                             num_workers=2, 
                             pin_memory=False)  # Disable pin_memory
    
    eval_dataset = test_dataset
    eval_dataloader_all = DataLoader(test_dataset, 
                                   batch_size=16,  # Reduce from 32
                                   shuffle=False,
                                   drop_last=False, 
                                   num_workers=2, 
                                   pin_memory=True)

    ##initialize model optimizers
    # After loading the pre-trained weights, freeze the backbone parameters
    # For source model
    for param in model_feature_source[0].parameters():  # backbone
        param.requires_grad = False

    # For target model
    for param in model_feature_target[0].parameters():  # backbone
        param.requires_grad = False

    # Enable gradients for bottleneck and pool layer
    for param in model_feature_source[1].parameters():  # pool layer
        param.requires_grad = True
    for param in model_feature_source[2].parameters():  # bottleneck
        param.requires_grad = True

    for param in model_feature_target[1].parameters():  # pool layer
        param.requires_grad = True
    for param in model_feature_target[2].parameters():  # bottleneck
        param.requires_grad = True

    # Different learning rates for different components
    feature_source_optim = torch.optim.SGD([
        {'params': model_feature_source[1].parameters()},  # pool layer
        {'params': model_feature_source[2].parameters()},  # bottleneck
    ], lr=lr, momentum=momentum)

    feature_target_optim = torch.optim.SGD([
        {'params': model_feature_target[1].parameters()},  # pool layer
        {'params': model_feature_target[2].parameters()},  # bottleneck
    ], lr=lr, momentum=momentum)

    clf_optim = torch.optim.SGD(model_clf.parameters(), lr=lr, momentum=momentum)


    # Modify the learning rate schedulers to include warmup
    def warmup_lr_scheduler(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0 / (1.0 + lr_gamma * (epoch - warmup_epochs) ** 0.5)  # Slower decay

    feature_source_scheduler = LambdaLR(feature_source_optim, warmup_lr_scheduler)
    feature_target_scheduler = LambdaLR(feature_target_optim, warmup_lr_scheduler)
    clf_scheduler = LambdaLR(clf_optim, warmup_lr_scheduler)
    
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

    # After calculating per-class totals
    print("\nPer-class sample counts:")
    for i in range(K):
        print(f"    {class_names[i]}: {per_class_total[i].item():.0f} samples")

    # When calculating accuracies
    print("\nPer-class correct predictions:")
    for i in range(K):
        print(f"    {class_names[i]}: {per_class_correct[i].item():.0f} correct out of {per_class_total[i].item():.0f}")
        
    # Modify accuracy calculation to handle zero division
    per_class_accuracies = torch.where(
        per_class_total > 0,
        per_class_correct / per_class_total,
        torch.tensor(float('nan'), device=device)
    )

    # After creating the models
    def set_checkpointing(model):
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(enable=True)
        elif hasattr(model, 'grad_checkpointing'):
            model.grad_checkpointing = True
        else:
            # For timm ViT models
            if hasattr(model, 'blocks'):
                for block in model.blocks:
                    if hasattr(block, 'attn'):
                        block.attn._checkpoint = True
                    if hasattr(block, 'mlp'):
                        block.mlp._checkpoint = True

    # Apply checkpointing to both models
    set_checkpointing(model_feature_source[0])
    set_checkpointing(model_feature_target[0])

    # At the start of training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Initialize mixed precision training
    # scaler = GradScaler(enabled=True)
    
    # Unfreeze the last few layers of the backbone
    def unfreeze_last_n_layers(model, n=4):
        if hasattr(model, 'blocks'):
            for i, block in enumerate(model.blocks):
                for param in block.parameters():
                    param.requires_grad = (i >= len(model.blocks) - n)

    # Unfreeze last few layers of both backbones
    unfreeze_last_n_layers(model_feature_source[0])
    unfreeze_last_n_layers(model_feature_target[0])

    # Inside the evaluation loop, add more detailed logging
    def evaluate_model(model_feature_target, model_clf, eval_dataloader, device):
        model_feature_target.eval()
        model_clf.eval()
        
        total_loss = 0
        predictions_all = []
        labels_all = []
        
        with torch.no_grad():
            for x, y_true in tqdm(eval_dataloader, leave=False):
                x, y_true = x.to(device), y_true.to(device)
                
                # Get predictions
                features = model_feature_target(x)
                logits = model_clf(features)
                predictions = logits.max(1)[1]
                
                # Store predictions and labels
                predictions_all.extend(predictions.cpu().numpy())
                labels_all.extend(y_true.cpu().numpy())
                
                # Calculate loss
                loss = F.cross_entropy(logits, y_true)
                total_loss += loss.item()
                
        return np.array(predictions_all), np.array(labels_all), total_loss / len(eval_dataloader)

    # At the start of each epoch
    def verify_model_state(model, name):
        print(f"\nVerifying {name} state:")
        requires_grad = any(p.requires_grad for p in model.parameters())
        has_grad = any(p.grad is not None for p in model.parameters())
        print(f"  Requires grad: {requires_grad}")
        print(f"  Has grad: {has_grad}")
        print(f"  Training mode: {model.training}")


    for epoch in range(1, epochs+1):
        print("epoch", epoch)

        # Make sure model is in train mode
        model_feature_source.train()
        model_feature_target.train()
        model_clf.train()

        # Add these calls at the start of each epoch
        #verify_model_state(model_feature_source, "Source Feature Extractor")
        #verify_model_state(model_feature_target, "Target Feature Extractor")
        #verify_model_state(model_clf, "Classifier")
        
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
        
        # For testing, only run a few iterations
        if True:  # Change to False for full training
            num_iterations = min(num_iterations,10)  # Only run 5 iterations per epoch
            print(f"Testing mode: Running only {num_iterations} iterations per epoch")
        
        # More gradual batch size increase
        current_batch = min(initial_batch * (1.2 ** (epoch - 1)), half_batch)  # Changed from 1.5 to 1.2
        current_batch = int(current_batch)
        print(f"Current batch size: {current_batch}")
        
        # Ensure minimum samples per class (approximately)
        min_samples_per_class = current_batch // K  # K is number of classes (12)
        print(f"Minimum samples per class (approx): {min_samples_per_class}")
        
        source_loader.batch_sampler.batch_size = current_batch
        target_loader.batch_sampler.batch_size = current_batch
        
        for i in tqdm(range(num_iterations)):
            # Clear memory before processing each batch
            torch.cuda.empty_cache()
            
            data_source = next(source_batch_iterator)
            source_x, source_y = data_source
            data_target = next(target_batch_iterator)
            target_x, _ = data_target
            
            # Move data to GPU in smaller chunks if needed
            source_x = source_x.to(device, non_blocking=True)
            target_x = target_x.to(device, non_blocking=True)
            source_y = source_y.to(torch.int64).to(device, non_blocking=True)
            
            # Regular forward pass without autocast
            source_feature = model_feature_source(source_x)
            target_feature = model_feature_target(target_x)
            target_feature_2 = target_feature.detach().clone()

            # Ensure source_preds requires grad
            source_preds = model_clf(source_feature)
            clf_loss_unweight = clf_criterion(source_preds, source_y)
            report_clf_loss_unweight = clf_criterion_unweight(source_preds, source_y)
            
            # Get target predictions
            target_logits = model_clf(target_feature_2)
            target_preds = torch.argmax(target_logits, 1)

            # Get clustering information
            cluster_s = F.one_hot(source_y, num_classes=K).float()
            cluster_t = F.one_hot(target_preds, num_classes=K).float()

            # Weighted classification loss
            weighted_clf_err = cluster_s * clf_loss_unweight.reshape(-1,1)
            expected_clf_err = torch.mean(weighted_clf_err, dim=0)
            clf_loss = torch.sum(expected_clf_err.reshape(K,1) * w_imp)

            # Weighted domain invariant loss   
            wasserstein_distance = torch.tensor(0., device=device, requires_grad=True)
            for cluster_id in range(K):
                if torch.sum(target_preds==cluster_id)!=0 and torch.sum(source_y==cluster_id)!=0:
                    source_cluster = source_feature[source_y==cluster_id]
                    target_cluster = target_feature[target_preds==cluster_id]
                    if len(source_cluster) > 0 and len(target_cluster) > 0:
                        cluster_distance = w1_loss(source_cluster, target_cluster)
                        wasserstein_distance = wasserstein_distance + (w_tgt[cluster_id] * cluster_distance)

            # Compute other losses with gradient tracking
            source_sntg_loss = sntg_loss_func(cluster=cluster_s, feature=source_feature, LAMBDA=LAMBDA)
            target_sntg_loss = sntg_loss_func(cluster=cluster_t, feature=target_feature, LAMBDA=LAMBDA)
            sntg_loss = (source_sntg_loss + target_sntg_loss).requires_grad_(True)

            centroid_loss = centroid_loss_func(K, device, source_y, target_preds, source_feature, target_feature)
            if not torch.isnan(centroid_loss):
                centroid_loss = centroid_loss.requires_grad_(True)

            # Ensure clf_loss has gradients
            clf_loss = clf_loss.requires_grad_(True)

            # Combine losses with explicit gradient requirements
            if torch.isnan(centroid_loss):
                loss = torch.zeros(1, device=device, requires_grad=True)
                loss = loss + lamb_clf * clf_loss
                loss = loss + lamb_wd * wasserstein_distance
                loss = loss + lamb_sntg * sntg_loss
            else:
                loss = torch.zeros(1, device=device, requires_grad=True)
                loss = loss + lamb_clf * clf_loss
                loss = loss + lamb_wd * wasserstein_distance
                loss = loss + lamb_centroid * centroid_loss
                loss = loss + lamb_sntg * sntg_loss

            # Verify loss requires grad before backward
            assert loss.requires_grad, "Loss doesn't require grad!"

            # Clear gradients
            feature_source_optim.zero_grad()
            feature_target_optim.zero_grad()
            clf_optim.zero_grad()
            
            # Combine losses with explicit gradient tracking
            loss = (lamb_clf * clf_loss + 
                   lamb_wd * wasserstein_distance + 
                   (lamb_centroid * centroid_loss if not torch.isnan(centroid_loss) else 0) + 
                   lamb_sntg * sntg_loss)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model_feature_source.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model_feature_target.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model_clf.parameters(), max_norm=1.0)
                        
            # Clear some memory
            del source_feature, target_feature, target_feature_2
            torch.cuda.empty_cache()
            
            #total_w1_loss += wasserstein_distance.item()
            total_unweight_clf_loss += report_clf_loss_unweight.item()
            total_clf_loss += clf_loss.item()
            if torch.isnan(centroid_loss) == False:
                total_centroid_loss += centroid_loss.item()
            total_sntg_loss += sntg_loss.item()
            total_w1_original += wasserstein_distance.item()
            
            # After backward pass, before optimizer steps
            def print_grad_norms():
                print("\nGradient norms:")
                for name, param in model_feature_source.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: {param.grad.norm().item()}")
                for name, param in model_feature_target.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: {param.grad.norm().item()}")
                for name, param in model_clf.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: {param.grad.norm().item()}")

            # Print gradients every N iterations
            #if i % 100 == 0:
            #    print_grad_norms()
            
            if i % 100 == 0:  # Every 100 iterations
                print(f"Iteration {i}: Loss={loss.item():.4f}, "
                      f"CLF={clf_loss.item():.4f}, "
                      f"W1={wasserstein_distance.item():.4f}, "
                      f"Centroid={centroid_loss.item():.4f}, "
                      f"SNTG={sntg_loss.item():.4f}")
            
            #if i % 100 == 0:  # Every 100 iterations
            #    print("\nGradient Statistics:")
            #    for name, param in model_feature_target.named_parameters():
            #        if param.requires_grad and param.grad is not None:
            #            grad_norm = param.grad.norm().item()
            #            weight_norm = param.norm().item()
            #            print(f"{name}: grad_norm={grad_norm:.6f}, weight_norm={weight_norm:.6f}")
            
            if i == 0:  # Only check first iteration
                print("\nChecking gradient flow:")
                for name, param in model_feature_target.named_parameters():
                    if param.requires_grad:
                        if param.grad is None:
                            print(f"{name} has no gradient!")
                        else:
                            grad_norm = param.grad.norm().item()
                            if grad_norm == 0:
                                print(f"{name} has zero gradient!")
            
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
        
        # Update weights
        feature_source_optim.step()
        feature_target_optim.step()
        clf_optim.step()     
                   
        ##evaluate models on target domain##
        if epoch % 1 == 0: 
            # Set to eval mode
            model_feature_source.eval()
            model_feature_target.eval()
            model_clf.eval()
        
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

        # Print current learning rates
        current_lr_source = feature_source_optim.param_groups[0]['lr']
        current_lr_target = feature_target_optim.param_groups[0]['lr']
        current_lr_clf = clf_optim.param_groups[0]['lr']
        print(f'Learning rates: source={current_lr_source:.6f}, target={current_lr_target:.6f}, clf={current_lr_clf:.6f}')

        # At the end of each epoch
        feature_source_scheduler.step()
        feature_target_scheduler.step()
        clf_scheduler.step()

        # After evaluation, add:
        model_feature_source.train()
        model_feature_target.train()
        model_clf.train()


# Remove indentation for the main block
if __name__ == '__main__':
    K = 12  # Number of classes in VisDA-2017
    batch_size = 256  # Smaller batch size
    lr = 0.01  # Increase learning rate
    lamb_clf = 1.0
    lamb_wd = 0.2      # Increase from 0.5
    lamb_centroid = 1.0  # Increase from 0.3
    lamb_sntg = 0.01     # Increase from 0.3
    LAMBDA = 30
    momentum = 0.9
    alpha = 8
    source_num = 152397  # Total number of synthetic images
    target_num = 55388   # Total number of real images
    eval_target_num = 5000  # Number of validation samples
    lr_gamma = 0.1
    lr_decay = 0.75

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
