import argparse
import sys
sys.path.insert(0, '../../')
import numpy as np
import torch
import pickle as pkl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
import torch.nn.functional as F
import utils.config as config
from models.model_neural import *
from utils.helper import *
sys.path.insert(0, '../../')

seed_torch(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(batch_size):
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
    num_b =  [6000,3000,6000] #[3000,6000,8000] 
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
    
    dataset = my_dataset(X_source, y_source)
    shuffled_indices = np.random.permutation(len(dataset))
    train_idx = shuffled_indices[:int(0.8*len(dataset))]
    val_idx = shuffled_indices[int(0.8*len(dataset)):]
    
    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=0, pin_memory=True)
    
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=0, pin_memory=True)
    
    return train_loader, val_loader


def do_epoch(model_feature,model_classifier, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    LAMBDA = 30
    K = 3
    
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
        
        
        loss = loss + 0.1*source_sntg_loss 

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
    seed_torch(666)
    train_loader, val_loader = create_dataloaders(args.batch_size)

    model_feature = DenseNet_bw(dim_in=616).to(device)
    model_classifier = Classifier(num_classes=3).to(device)
    optim = torch.optim.Adam(list(model_feature.parameters()) + list(model_classifier.parameters()))
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

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(model_feature.state_dict(), 'trained_models/source_feature_b_to_w.pt')
            torch.save(model_classifier.state_dict(), 'trained_models/source_clf_b_to_w.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on Neural')
    arg_parser.add_argument('--batch-size', type=int, default=128)
    arg_parser.add_argument('--epochs', type=int, default=200)
    args = arg_parser.parse_args()
    main(args)
