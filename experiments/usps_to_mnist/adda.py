"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
"""
import argparse
import sys
sys.path.insert(0, '../../')
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm, trange
import utils.config as config
from utils.helper import *
from models.model_DANN import *
from data.mnist_mnistm_data import *
from data.usps import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    seed_torch(seed=123)
    source_model = Net().to(device)
    source_model.load_state_dict(torch.load('trained_models/DANN_source.pt'))
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    
    clf = source_model
    source_model = source_model.feature_extractor

    target_model = Net().to(device)
    target_model.load_state_dict(torch.load('trained_models/DANN_source.pt'))
    target_model = target_model.feature_extractor

    discriminator = nn.Sequential(
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    source_dataset = load_USPS_LS(target_num=180,train_flag=True)
    source_loader = DataLoader(source_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=0, pin_memory=True)
    
    target_dataset = load_mnist_LS(source_num=290,train_flag=False)
    target_loader = DataLoader(target_dataset, batch_size=half_batch,
                               shuffle=True, num_workers=0, pin_memory=True)

    discriminator_optim = torch.optim.Adam(discriminator.parameters(),lr=0.001)
    target_optim = torch.optim.Adam(target_model.parameters(),lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    best_target_acc = 0
    for epoch in range(1, args.epochs+1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        total_loss = 0
        total_accuracy = 0
        for _ in trange(args.iterations, leave=False):
            # Train discriminator
            set_requires_grad(target_model, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(args.k_disc):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)

                source_features = source_model(source_x).view(source_x.shape[0], -1)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                             torch.zeros(target_x.shape[0], device=device)])

                preds = discriminator(discriminator_x).squeeze()
                loss = criterion(preds, discriminator_y)

                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()

                total_loss += loss.item()
                total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()

            # Train classifier
            set_requires_grad(target_model, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(args.k_clf):
                _, (target_x, _) = next(batch_iterator)
                target_x = target_x.to(device)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)

                preds = discriminator(target_features).squeeze()
                loss = criterion(preds, discriminator_y)

                target_optim.zero_grad()
                loss.backward()
                target_optim.step()

        mean_loss = total_loss / (args.iterations*args.k_disc)
        mean_accuracy = total_accuracy / (args.iterations*args.k_disc)
        tqdm.write(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, '
                   f'discriminator_accuracy={mean_accuracy:.4f}')

        # Create the full target model and save it
        clf.feature_extractor = target_model
        #torch.save(clf.state_dict(), 'trained_models/adda.pt')
        
        dataset_test = load_mnist_LS(source_num=290,train_flag=False)
        dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False,
                                drop_last=False, num_workers=0, pin_memory=True)
        clf = clf.eval()
        total_accuracy = 0
        with torch.no_grad():
            for x, y_true in tqdm(dataloader_test, leave=False):
                x, y_true = x.to(device), y_true.to(device)
                y_pred = clf(x)
                total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()

        mean_accuracy = total_accuracy / len(dataloader_test)
        if mean_accuracy > best_target_acc:
            best_target_acc = mean_accuracy
            torch.save(clf.state_dict(), 'trained_models/adda.pt')
        print(f'Accuracy on target data: {mean_accuracy:.4f}')
        print(f'Current best accuracy on target data: {best_target_acc:.4f}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using ADDA')
    arg_parser.add_argument('--batch-size', type=int, default=128)
    arg_parser.add_argument('--iterations', type=int, default=500)
    arg_parser.add_argument('--epochs', type=int, default=500)
    arg_parser.add_argument('--k-disc', type=int, default=1)
    arg_parser.add_argument('--k-clf', type=int, default=1)
    args = arg_parser.parse_args()
    main(args)
