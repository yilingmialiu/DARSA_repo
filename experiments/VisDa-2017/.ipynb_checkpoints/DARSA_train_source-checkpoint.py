import argparse
import sys
sys.path.append('/datacommons/carlsonlab/yl407/packages')
sys.path.insert(0, '../../')
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
import torch.nn.functional as F
import utils.config as config
from utils.helper import *
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
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau

seed_torch(123) # 0 #123
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(batch_size):
    train_resizing = 'default'
    val_resizing = 'default'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    dataset = train_source_dataset
    shuffled_indices = np.random.permutation(len(dataset))
    train_idx = shuffled_indices[:int(0.8*len(dataset))]
    val_idx = shuffled_indices[int(0.8*len(dataset)):]

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=1, pin_memory=True)

    target_dataset_label = val_dataset
    target_loader = DataLoader(target_dataset_label, batch_size=batch_size,drop_last=True,\
                               shuffle=True, num_workers=1, pin_memory=True)
    return train_loader, val_loader, target_loader


def do_epoch(model_feature,model_classifier, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    LAMBDA = 30
    K = 65
    
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


def main(args):
    train_loader, val_loader,target_loader = create_dataloaders(args.batch_size)
    backbone = get_model('vit_base_patch16_224', pretrain=True)
    pool_layer = nn.Identity()
    model_feature = Net(backbone, args.num_classes, bottleneck_dim=256,
                        pool_layer=pool_layer, finetune=False).to(device)
    model_classifier = Classifier().to(device)
    optim = torch.optim.SGD(list(model_feature.parameters()) + list(model_classifier.parameters()),lr=1e-3,\
                            momentum=0.99) #0.99
    #optim = torch.optim.Adam(list(model_feature.parameters()) + list(model_classifier.parameters()))
    scheduler = StepLR(optim, step_size=20, gamma=0.1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, args.epochs+1):
        model_feature = model_feature.train()
        model_classifier = model_classifier.train()
        train_loss, train_accuracy = do_epoch(model_feature,model_classifier,train_loader, criterion, optim=optim)

        model_feature = model_feature.eval()
        model_classifier = model_classifier.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model_feature,model_classifier, val_loader, criterion, optim=None)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        with torch.no_grad():
            target_loss, target_accuracy = do_epoch(model_feature,model_classifier,target_loader, criterion, optim=None)
            print("target_accuracy",target_accuracy)
            #if val_accuracy > best_accuracy:
            if target_accuracy > best_accuracy:
                print('Saving model...')
                #best_accuracy = val_accuracy
                best_accuracy = target_accuracy
                torch.save(model_feature.state_dict(), 'trained_models/source_feature_rd_128_SGD_sntg_0.01.pt')
                torch.save(model_classifier.state_dict(), 'trained_models/source_clf_rd_128_SGD_sntg_0.01.pt')
            
        scheduler.step()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on OH')
    arg_parser.add_argument('--batch-size', type=int, default=64)#64
    arg_parser.add_argument('--epochs', type=int, default=100) #40
    arg_parser.add_argument('--num-classes', type=int, default=65)
    args = arg_parser.parse_args()
    main(args)
