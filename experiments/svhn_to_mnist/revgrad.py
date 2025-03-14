"""
Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
"""
import argparse
import sys
sys.path.insert(0, '../../')
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from data.mnist_mnistm_data import *
from data.svhn import *
import utils.config as config
from utils.helper import *
from models.model_DANN import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    seed_torch(seed=123)
    model = Net().to(device)
    model.load_state_dict(torch.load('trained_models/DANN_source.pt'))
    feature_extractor = model.feature_extractor
    clf = model.classifier
    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    source_dataset = load_SVHN_LS(target_num=1500,train_flag='train')
    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=0, pin_memory=True)
    
    target_dataset = load_mnist_LS(source_num=290,train_flag=False)
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=0, pin_memory=True)

    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()),lr=1e-4)
    best_target_acc = 0
    for epoch in range(1, args.epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_domain_loss = total_label_accuracy = 0
        for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
                x = torch.cat([source_x, target_x])
                x = x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)

                features = feature_extractor(x).view(x.shape[0], -1)
                domain_preds = discriminator(features).squeeze()
                label_preds = clf(features[:source_x.shape[0]])
                
                domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
                label_loss = F.cross_entropy(label_preds, label_y)
                loss = domain_loss + label_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_domain_loss += domain_loss.item()
                total_label_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}')
        
        dataset_test = load_mnist_LS(source_num=290,train_flag=False)
        dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False,
                                drop_last=False, num_workers=0, pin_memory=True)
        model = model.eval()
        total_accuracy = 0
        with torch.no_grad():
            for x, y_true in tqdm(dataloader_test, leave=False):
                x, y_true = x.to(device), y_true.to(device)
                y_pred = model(x)
                total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()

        mean_accuracy = total_accuracy / len(dataloader_test)
        if mean_accuracy > best_target_acc:
            best_target_acc = mean_accuracy
            torch.save(model.state_dict(), 'trained_models/DANN.pt')
        print(f'Accuracy on target data: {mean_accuracy:.4f}')
        print(f'Current best accuracy on target data: {best_target_acc:.4f}')
        

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
    arg_parser.add_argument('--batch-size', type=int, default=128)
    arg_parser.add_argument('--epochs', type=int, default=300)
    args = arg_parser.parse_args()
    main(args)
