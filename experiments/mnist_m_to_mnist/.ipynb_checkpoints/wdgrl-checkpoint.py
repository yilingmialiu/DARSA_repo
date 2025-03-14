"""
Implements WDGRL:
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""
import argparse
import sys
sys.path.insert(0, '../../')
import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm, trange
import utils.config as config
from utils.helper import *
from models.model_DANN import *
from data.mnist_mnistm_data import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty


def main(args):
    seed_torch(seed=123)
    clf_model = Net().to(device)
    clf_model.load_state_dict(torch.load('trained_models/DANN_source.pt'))
    
    feature_extractor = clf_model.feature_extractor
    discriminator = clf_model.classifier

    critic = nn.Sequential(
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    source_dataset = load_mnistm_LS(target_num=1800,MNISTM=MNISTM,train_flag=True)
    source_loader = DataLoader(source_dataset, batch_size=half_batch, drop_last=True,
                               shuffle=True, num_workers=0, pin_memory=True)
    
    target_dataset = load_mnist_LS(source_num=290,train_flag=False)
    target_loader = DataLoader(target_dataset, batch_size=half_batch, drop_last=True,
                               shuffle=True, num_workers=0, pin_memory=True)

    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-4)
    clf_optim = torch.optim.Adam(clf_model.parameters(), lr=1e-4)
    clf_criterion = nn.CrossEntropyLoss()
    best_target_acc = 0

    for epoch in range(1, args.epochs+1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        total_loss = 0
        total_accuracy = 0
        for _ in trange(args.iterations, leave=False):
            (source_x, source_y), (target_x, _) = next(batch_iterator)
            # Train critic
            set_requires_grad(feature_extractor, requires_grad=False)
            set_requires_grad(critic, requires_grad=True)

            source_x, target_x = source_x.to(device), target_x.to(device)
            source_y = source_y.to(device)

            with torch.no_grad():
                h_s = feature_extractor(source_x).data.view(source_x.shape[0], -1)
                h_t = feature_extractor(target_x).data.view(target_x.shape[0], -1)
            for _ in range(args.k_critic):
                gp = gradient_penalty(critic, h_s, h_t)

                critic_s = critic(h_s)
                critic_t = critic(h_t)
                wasserstein_distance = critic_s.mean() - critic_t.mean()

                critic_cost = -wasserstein_distance + args.gamma*gp

                critic_optim.zero_grad()
                critic_cost.backward()
                critic_optim.step()

                total_loss += critic_cost.item()

            # Train classifier
            set_requires_grad(feature_extractor, requires_grad=True)
            set_requires_grad(critic, requires_grad=False)
            for _ in range(args.k_clf):
                source_features = feature_extractor(source_x).view(source_x.shape[0], -1)
                target_features = feature_extractor(target_x).view(target_x.shape[0], -1)

                source_preds = discriminator(source_features)
                clf_loss = clf_criterion(source_preds, source_y)
                wasserstein_distance = critic(source_features).mean() - critic(target_features).mean()

                loss = clf_loss + args.wd_clf * wasserstein_distance
                clf_optim.zero_grad()
                loss.backward()
                clf_optim.step()

        mean_loss = total_loss / (args.iterations * args.k_critic)
        tqdm.write(f'EPOCH {epoch:03d}: critic_loss={mean_loss:.4f}')
        #torch.save(clf_model.state_dict(), 'trained_models/wdgrl.pt')
        dataset_test = load_mnist_LS(source_num=290,train_flag=False)
        
        dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False,
                                drop_last=False, num_workers=0, pin_memory=True)
        clf_model = clf_model.eval()
        total_accuracy = 0
        with torch.no_grad():
            for x, y_true in tqdm(dataloader_test, leave=False):
                x, y_true = x.to(device), y_true.to(device)
                y_pred = clf_model(x)
                total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()

        mean_accuracy = total_accuracy / len(dataloader_test)
        if mean_accuracy > best_target_acc:
            best_target_acc = mean_accuracy
            torch.save(clf_model.state_dict(), 'trained_models/wdgrl.pt')
        print(f'Accuracy on target data: {mean_accuracy:.4f}')
        print(f'Current best accuracy on target data: {best_target_acc:.4f}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using WDGRL')
    arg_parser.add_argument('--batch-size', type=int, default=128)
    arg_parser.add_argument('--iterations', type=int, default=500)
    arg_parser.add_argument('--epochs', type=int, default=100)
    arg_parser.add_argument('--k-critic', type=int, default=5)
    arg_parser.add_argument('--k-clf', type=int, default=10)
    arg_parser.add_argument('--gamma', type=float, default=10)
    arg_parser.add_argument('--wd-clf', type=float, default=0.1)
    args = arg_parser.parse_args()
    main(args)
