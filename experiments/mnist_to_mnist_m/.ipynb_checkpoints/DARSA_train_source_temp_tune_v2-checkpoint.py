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
from models.model_mnist_mnistm import *
from data.mnist_mnistm_data import *
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau

seed_new = 123 #39,52,right now at 62
seed_torch(seed_new) #15  #6
print("seed:",seed_new)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(batch_size,alpha):
    dataset = load_mnist_LS(source_num=600,alpha=alpha,train_flag=True)
    shuffled_indices = np.random.permutation(len(dataset))
    train_idx = shuffled_indices[:int(0.8*len(dataset))]
    val_idx = shuffled_indices[int(0.8*len(dataset)):]

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=1, pin_memory=True)
    
    target_dataset_label = load_mnistm_LS(target_num=10,alpha=alpha,MNISTM=MNISTM,train_flag=False)
    target_loader = DataLoader(target_dataset_label, batch_size=len(target_dataset_label),drop_last=True,\
                               shuffle=True, num_workers=1, pin_memory=True)
    
    return train_loader, val_loader, target_loader


def do_epoch(model_feature,model_classifier, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    LAMBDA = 30 
    K = 10
    
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
        
        
        loss = loss + source_sntg_loss

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
    train_loader, val_loader,target_loader = create_dataloaders(args.batch_size,args.alpha)

    model_feature = Net().to(device)
    model_classifier = Classifier().to(device)
    optim = torch.optim.SGD(list(model_feature.parameters()) + list(model_classifier.parameters()),lr=1e-3,momentum=0.99)
    scheduler = StepLR(optim, step_size=30, gamma=0.1, verbose=True) #30
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
            #if val_accuracy > best_accuracy:
            if val_accuracy > best_accuracy:
                print('Saving model...')
                best_accuracy = val_accuracy
                torch.save(model_feature.state_dict(), 'trained_models/source_feature_rd_128_SGD_alpha_'+str(args.alpha)+'.pt')
                torch.save(model_classifier.state_dict(), 'trained_models/source_clf_rd_128_SGD_alpha_'+str(args.alpha)+'.pt')
        
        scheduler.step()
    
    ##initialize models
    #with torch.no_grad():
        #model_feature = Net().to(device)
        #model_classifier = Classifier().to(device)
        #path_to_model = 'trained_models/'
        #model_feature.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_alpha_'+str(args.alpha)+'_v2.pt'))
        #model_feature_target.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_alpha_'+str(args.alpha)+'_v2.pt'))
        #model_classifier.load_state_dict(torch.load(path_to_model+'source_clf_rd_128_SGD_alpha_'+str(args.alpha)+'_v2.pt'))
        #target_loss, target_accuracy = do_epoch(model_feature,model_classifier,target_loader, criterion, optim=None)
        #print("load again target_accuracy",target_accuracy)
    
    
    
        #if val_accuracy > best_accuracy:
        #    print('Saving model...')
        #    best_accuracy = val_accuracy
        #    torch.save(model_feature.state_dict(), 'trained_models/source_feature_rd_128_SGD_alpha_'+str(args.alpha)+'.pt')
        #    torch.save(model_classifier.state_dict(), 'trained_models/source_clf_rd_128_SGD_alpha_'+str(args.alpha)+'.pt')
        #with torch.no_grad():
        #    if val_accuracy > best_accuracy:
        #        print('Saving model...')
        #        best_accuracy = val_accuracy
                
        #    if epoch > 10:
        #        model_feature = model_feature.eval()
        #        model_classifier = model_classifier.eval()
        #        target_loss, target_accuracy = do_epoch(model_feature,model_classifier,target_loader, criterion, optim=None)
        #        print("target_accuracy",target_accuracy)
                
        #        torch.save(model_feature.state_dict(), 'trained_models/source_feature_rd_128_SGD_alpha_'+str(args.alpha)+'_v2.pt')
        #        torch.save(model_classifier.state_dict(), 'trained_models/source_clf_rd_128_SGD_alpha_'+str(args.alpha)+'_v2.pt')
                
        #scheduler.step()
                
        

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on MNIST')
    arg_parser.add_argument('--batch-size', type=int, default=256) #256
    arg_parser.add_argument('--epochs', type=int, default=20) #30
    arg_parser.add_argument('--alpha', type=int, default=8)
    args = arg_parser.parse_args()
    main(args)
