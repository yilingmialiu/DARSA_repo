import random
import os
import sys
sys.path.insert(0, '../../')
sys.path.append('/datacommons/carlsonlab/yl407/packages')
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.model_CDAN import *
from utils.helperCDAN import *
from data.mnist_mnistm_data import *
from utils.helper import *

seed_torch(seed=123)

def train(args, model, ad_net, random_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch, start_epoch, method):
    seed_torch(seed=123)
    model.train()
    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target
    
    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)    
        if batch_idx % len_target == 0:
            iter_target = iter(train_loader1)
        data_source, label_source = next(iter_source)
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target = next(iter_target)
        data_target = data_target.cuda()
        optimizer.zero_grad()
        optimizer_ad.zero_grad()
        feature, output = model(torch.cat((data_source, data_target), 0))
        loss = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)
        softmax_output = nn.Softmax(dim=1)(output)
        if epoch > start_epoch:
            if method == 'CDAN-E':
                entropy = Entropy(softmax_output)
                loss += CDAN([feature, softmax_output], ad_net, entropy, calc_coeff(num_iter*(epoch-start_epoch)+batch_idx), random_layer)
            elif method == 'CDAN':
                loss += CDAN([feature, softmax_output], ad_net, None, None, random_layer)
            elif method == 'DANN':
                loss += DANN(feature, ad_net)
            else:
                raise ValueError('Method cannot be recognized.')
        loss.backward()
        optimizer.step()
        if epoch > start_epoch:
            optimizer_ad.step()
        if (batch_idx+epoch*num_iter) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*args.batch_size, num_iter*args.batch_size,
                100. * batch_idx / num_iter, loss.item()))

def test(args, model, test_loader):
    seed_torch(seed=123)
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            feature, output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.data.cpu().max(1, keepdim=True)[1]
            correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CDAN MNIST MNISTM')
    parser.add_argument('--method', type=str, default='CDAN-E', choices=['CDAN', 'CDAN-E', 'DANN'])
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu_id', type=str,
                        help='cuda device id')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--random', type=bool, default=False,
                        help='whether to use random')
    args = parser.parse_args()

    #torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


    dataset_source = load_mnist_LS(source_num=600,alpha=8,train_flag=True)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    dataset_target = load_mnistm_LS(target_num=100,alpha=8,MNISTM=MNISTM,train_flag=True)
    
    train_loader1 = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    #evaluation on a subset of the test
    dataset_test = load_mnistm_LS(target_num=100,alpha=8,MNISTM=MNISTM,train_flag=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, 
                                                  batch_size=len(dataset_test), 
                                                  shuffle=False,drop_last=False, 
                                                  num_workers=0, pin_memory=True)
    

    model = DTN()
    model = model.cuda()
    class_num = 10

    if args.random:
        random_layer = RandomLayer([model.output_num(), class_num], 500)
        ad_net = AdversarialNetwork(500, 500)
        random_layer.cuda()
    else:
        random_layer = None
        ad_net = AdversarialNetwork(model.output_num() * class_num, 500)
    ad_net = ad_net.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=args.momentum)
    optimizer_ad = optim.SGD(ad_net.parameters(), lr=args.lr, weight_decay=0.0005, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        if epoch % 3 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.3
        train(args, model, ad_net, random_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch, 0, args.method)
        test(args, model, test_loader)

if __name__ == '__main__':
    main()
