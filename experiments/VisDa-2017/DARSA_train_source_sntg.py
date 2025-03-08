import argparse
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, project_root)

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import random
import time
import warnings
import shutil
import os.path as osp
from utils.helper import *
from utils.helperVisDa import *
from data.visda2017 import VisDA2017
from models.model_visda import *
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
from utils.helper import seed_torch
seed_torch(123) # 0 #123
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(batch_size):
    train_resizing = 'default'
    val_resizing = 'default'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names = \
    get_dataset(dataset_name='VisDA2017', 
                root='/data/home/yilingliu/VisDA-2017',
                source={'Synthetic': 'image_list/train_list_imbalanced.txt'}, 
                target={'Real': 'image_list/validation_list_imbalanced.txt'}, 
                train_source_transform=train_transform, 
                val_transform=val_transform)
    
    print("Source dataset size:", len(train_source_dataset))
    print("Target dataset size:", len(train_target_dataset))
    
    dataset = train_source_dataset
    shuffled_indices = np.random.permutation(len(dataset))
    train_idx = shuffled_indices[:int(0.8*len(dataset))]
    val_idx = shuffled_indices[int(0.8*len(dataset)):]

    train_loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            drop_last=True,
                            sampler=SubsetRandomSampler(train_idx),
                            num_workers=2, 
                            pin_memory=True)
    
    val_loader = DataLoader(dataset, 
                          batch_size=batch_size, 
                          drop_last=False,
                          sampler=SubsetRandomSampler(val_idx),
                          num_workers=2, 
                          pin_memory=True)

    target_loader = DataLoader(test_dataset, 
                             batch_size=batch_size,
                             drop_last=True,
                             shuffle=True, 
                             num_workers=2, 
                             pin_memory=True)
    
    return train_loader, val_loader, target_loader


def do_epoch(model_feature,model_classifier, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    LAMBDA = 30
    K = 12
    
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        source_feature = model_feature(x)
        y_pred = model_classifier(source_feature)
        loss = criterion(y_pred, y_true)
        source_preds = torch.argmax(y_pred,1)
        source_y = source_preds.to(torch.int64).to(device)
        cluster_s = F.one_hot(source_y, num_classes=K).float()
        
        graph_source = torch.sum(cluster_s[:, None, :] * cluster_s[None, :, :], 2)
        distance_source = torch.mean((source_feature[:, None, :] - source_feature[None, :, :])**2, 2)
        source_sntg_loss = torch.mean(graph_source * distance_source + \
                                      (1-graph_source)*torch.nn.functional.relu(LAMBDA- distance_source))
        
        
        loss = loss + 0.01*source_sntg_loss #0.01

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy


def evaluate_target(model_feature, model_classifier, target_loader, criterion):
    model_feature.eval()
    model_classifier.eval()
    
    total_loss = 0
    per_class_correct = torch.zeros(12).to(device)  # VisDA has 12 classes
    per_class_total = torch.zeros(12).to(device)
    
    with torch.no_grad():
        for x, y_true in tqdm(target_loader, leave=False, desc='Evaluating on target'):
            x, y_true = x.to(device), y_true.to(device)
            
            # Forward pass
            features = model_feature(x)
            y_pred = model_classifier(features)
            loss = criterion(y_pred, y_true)
            
            # Calculate per-class accuracy
            predictions = y_pred.max(1)[1]
            for label in range(12):
                label_mask = (y_true == label)
                per_class_correct[label] += (predictions[label_mask] == label).sum()
                per_class_total[label] += label_mask.sum()
            
            total_loss += loss.item()
    
    # Calculate accuracies
    per_class_accuracies = (per_class_correct / (per_class_total + 1e-10)).cpu().numpy()
    mean_accuracy = per_class_accuracies.mean()
    mean_loss = total_loss / len(target_loader)
    
    # Print detailed results
    class_names = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
                  'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
    
    print("\nPer-class accuracy on target domain:")
    for i, (class_name, accuracy) in enumerate(zip(class_names, per_class_accuracies)):
        print(f"{class_name:12s}: {accuracy*100:.2f}%")
    print(f"\nMean accuracy: {mean_accuracy*100:.2f}%")
    
    return mean_loss, mean_accuracy, per_class_accuracies


def main(args):
    train_loader, val_loader, target_loader = create_dataloaders(args.batch_size)
    
    # Create the full model first
    backbone = get_model('vit_base_patch16_224', pretrain=True)
    full_model = ImageClassifier(backbone, args.num_classes, bottleneck_dim=256,
                               pool_layer=nn.Identity(), finetune=False).to(device)
    
    # Split the model into feature extractor and classifier
    model_feature = nn.Sequential(
        full_model.backbone,
        full_model.pool_layer,
        full_model.bottleneck
    ).to(device)
    
    model_classifier = full_model.head.to(device)
    
    # Rest of training remains the same
    optim = torch.optim.SGD(list(model_feature.parameters()) + list(model_classifier.parameters()),
                           lr=args.lr,
                           momentum=0.9,
                           weight_decay=0.001)
    
    scheduler = LambdaLR(optim, 
                        lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, args.epochs+1):
        # Training
        model_feature.train()
        model_classifier.train()
        train_loss, train_accuracy = do_epoch(model_feature, model_classifier, train_loader, criterion, optim=optim)

        # Validation on source
        model_feature.eval()
        model_classifier.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model_feature, model_classifier, val_loader, criterion, optim=None)

        # Evaluation on target
        target_loss, target_accuracy, per_class_accuracies = evaluate_target(
            model_feature, model_classifier, target_loader, criterion
        )

        tqdm.write(
            f'EPOCH {epoch:03d}: '
            f'train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f}, '
            f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}, '
            f'target_accuracy={target_accuracy:.4f}'
        )

        if target_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = target_accuracy
            torch.save({
                'epoch': epoch,
                'feature_state_dict': model_feature.state_dict(),
                'classifier_state_dict': model_classifier.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'best_accuracy': best_accuracy,
                'per_class_accuracies': per_class_accuracies
            }, f'trained_models/visda_source_model_best_sntg_0.01.pth')
            
        scheduler.step()

    # Load best model and evaluate
    print("\nEvaluating best model on target domain...")
    checkpoint = torch.load('trained_models/visda_source_model_best_sntg_0.01.pth')
    model_feature.load_state_dict(checkpoint['feature_state_dict'])
    model_classifier.load_state_dict(checkpoint['classifier_state_dict'])
    _, final_accuracy, final_per_class = evaluate_target(
        model_feature, model_classifier, target_loader, criterion
    )
    print(f"\nBest model's target accuracy: {final_accuracy*100:.2f}%")

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on VisDA-2017')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=100)
    arg_parser.add_argument('--num-classes', type=int, default=12)  # VisDA has 12 classes
    arg_parser.add_argument('--lr', type=float, default=0.01)
    arg_parser.add_argument('--lr-gamma', type=float, default=0.001,
                          help='Learning rate gamma for lambda scheduler')
    arg_parser.add_argument('--lr-decay', type=float, default=0.75,
                          help='Learning rate decay power for lambda scheduler')
    args = arg_parser.parse_args()
    main(args)
