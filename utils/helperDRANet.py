from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch


class _BatchInstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchInstanceNorm, self).__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        if self.affine:
            bn_w = self.weight * self.gate
        else:
            bn_w = self.gate
        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, bn_w, self.bias,
            self.training, self.momentum, self.eps)

        # Instance norm
        b, c = input.size(0), input.size(1)
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(
            input, None, None, None, None,
            True, self.momentum, self.eps)
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])

        return out_bn + out_in


class BatchInstanceNorm1d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm2d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm3d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
            
            
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import logging
import random
from PIL import Image


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # init.normal(m.weight, std=1e-3)
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def slice_patches(imgs, hight_slice=2, width_slice=4):
    b, c, h, w = imgs.size()
    h_patch, w_patch = int(h / hight_slice), int(w / width_slice)
    patches = imgs.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)
    patches = patches.contiguous().view(b, c, -1, h_patch, w_patch)
    patches = patches.transpose(1,2)
    patches = patches.reshape(-1, c, h_patch, w_patch)
    return patches

class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_weights(task, dsets):
    '''
        You must set these hyperparameters to apply our method to other datasets.
        These hyperparameters may not be the optimal value for your machine.
    '''

    alpha = dict()
    alpha['style'], alpha['dis'], alpha['gen'] = dict(), dict(), dict()
    if task == 'clf':
        alpha['recon'], alpha['consis'], alpha['content'] = 5, 1, 1

        # MNIST <-> MNIST-M
        if 'M' in dsets and 'MM' in dsets and 'U' not in dsets:
            alpha['style']['M2MM'], alpha['style']['MM2M'] = 5e4, 1e4
            alpha['dis']['M'], alpha['dis']['MM'] = 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['MM'] = 0.5, 1.0

        # MNIST <-> USPS
        elif 'M' in dsets and 'U' in dsets and 'MM' not in dsets:
            alpha['style']['M2U'], alpha['style']['U2M'] = 5e3, 5e3
            alpha['dis']['M'], alpha['dis']['U'] = 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['U'] = 0.5, 0.5
            
        # MNIST <-> USPS
        elif 'M' in dsets and 'S' in dsets and 'MM' not in dsets:
            alpha['style']['S2M'], alpha['style']['M2S'] = 5e3, 5e3
            alpha['dis']['S'], alpha['dis']['M'] = 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['S'] = 0.5, 0.5

        # MNIST <-> MNIST-M <-> USPS
        elif 'M' in dsets and 'U' in dsets and 'MM' in dsets:
            alpha['style']['M2MM'], alpha['style']['MM2M'], alpha['style']['M2U'], alpha['style']['U2M'] = 5e4, 1e4, 1e4, 1e4
            alpha['dis']['M'], alpha['dis']['MM'], alpha['dis']['U'] = 0.5, 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['MM'], alpha['gen']['U'] = 0.5, 1.0, 0.5

    elif task == 'seg':
        # GTA5 <-> Cityscapes
        alpha['recon'], alpha['consis'], alpha['content'] = 10, 1, 1
        alpha['style']['G2C'], alpha['style']['C2G'] = 5e3, 5e3
        alpha['dis']['G'], alpha['dis']['C'] = 0.5, 0.5
        alpha['gen']['G'], alpha['gen']['C'] = 0.5, 0.5

    return alpha


class Loss_Functions:
    def __init__(self, args):
        self.args = args
        self.alpha = loss_weights(args.task, args.datasets)

    def recon(self, imgs, recon_imgs):
        recon_loss = 0
        for dset in imgs.keys():
            recon_loss += F.l1_loss(imgs[dset], recon_imgs[dset])
        return self.alpha['recon'] * recon_loss
        
    def dis(self, real, fake):
        dis_loss = 0
        if self.args.task == 'clf':  # DCGAN loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.binary_cross_entropy(real[dset], torch.ones_like(real[dset]))
            for cv in fake.keys():
                source, target = cv.split('2')
                dis_loss += self.alpha['dis'][target] * F.binary_cross_entropy(fake[cv], torch.zeros_like(fake[cv]))
        elif self.args.task == 'seg':  # Hinge loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.relu(1. - real[dset]).mean()
            for cv in fake.keys():
                source, target = cv.split('2')
                dis_loss += self.alpha['dis'][target] * F.relu(1. + fake[cv]).mean()
        return dis_loss

    def gen(self, fake):
        gen_loss = 0
        for cv in fake.keys():
            source, target = cv.split('2')
            if self.args.task == 'clf':
                gen_loss += self.alpha['gen'][target] * F.binary_cross_entropy(fake[cv], torch.ones_like(fake[cv]))
            elif self.args.task == 'seg':
                gen_loss += -self.alpha['gen'][target] * fake[cv].mean()
        return gen_loss

    def content_perceptual(self, perceptual, perceptual_converted):
        content_perceptual_loss = 0
        for cv in perceptual_converted.keys():
            source, target = cv.split('2')
            content_perceptual_loss += F.mse_loss(perceptual[source][-1], perceptual_converted[cv][-1])
        return self.alpha['content'] * content_perceptual_loss

    def style_perceptual(self, style_gram, style_gram_converted):
        style_percptual_loss = 0
        for cv in style_gram_converted.keys():
            source, target = cv.split('2')
            for gr in range(len(style_gram[target])):
                style_percptual_loss += self.alpha['style'][cv] * F.mse_loss(style_gram[target][gr], style_gram_converted[cv][gr])
        return style_percptual_loss

    def consistency(self, contents, styles, contents_converted, styles_converted, converts):
        consistency_loss = 0
        for cv in converts:
            source, target = cv.split('2')
            consistency_loss += F.l1_loss(contents[cv], contents_converted[cv])
            consistency_loss += F.l1_loss(styles[target], styles_converted[cv])
        return self.alpha['consis'] * consistency_loss

    def task(self, pred, gt):
        task_loss = 0
        for key in pred.keys():
            if '2' in key:
                source, target = key.split('2')
            else:
                source = key
            task_loss += F.cross_entropy(pred[key], gt[source], ignore_index=-1)
        return task_loss

    
import numpy as np


def MIOU(GT, Pred, num_class=19):
    confusion_matrix = np.zeros((num_class,) * 2)
    mask = (GT >= 0) & (GT < num_class)
    label = num_class * GT[mask].astype('int') + Pred[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix += count.reshape(num_class, num_class)
    return confusion_matrix


def MIOU_score(GT, Pred, num_class=19):
    confusion_matrix = MIOU(GT, Pred, num_class)
    score = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
    score = np.nanmean(score)
    return score

import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
sys.path.insert(0, '../../')
## Import Data Loaders ##
from data.mnist_mnistm_data import *
from torch.nn.utils import spectral_norm
from models.model_mnist_mnistm import *
import utils.config as config
from utils.helper import *
from torchvision.transforms import Compose, ToTensor
from torchvision import datasets

class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)
    
import torch
import torch.nn as nn


def gram(x):
    (b, c, h, w) = x.size()
    f = x.view(b, c, h*w)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (c*w*h)
    return G


# def gram_content(x, y):
#     s1 = nn.Softmax(dim=1)
#     s2 = nn.Softmax(dim=0)
#     (b, c, h, w) = x.size()
#     fx = x.view(b, c*h*w)
#     fy = y.view(b, c*h*w)
#     fy = fy.transpose(0, 1)
#     G = fx @ fy
#     G = G / torch.norm(G, p=2)
#     G1 = s1(G)
#     G2 = s2(G)
#     return G1, G2


def cadt(source, target, style, W=10):
    s1 = nn.Softmax(dim=1)
    (b, c, h, w) = source.size()
    fs = source.view(b, c * h * w)
    ft = target.view(b, c * h * w)
    ft = ft.transpose(0, 1)
    H = fs @ ft
    H = H / torch.norm(H, p=2)
    H = s1(W * H)
    adaptive_style = style.view(b, c * h * w)
    adaptive_style = H @ adaptive_style
    adaptive_style = adaptive_style.view(b, c, h, w)
    # print(H)
    return H, adaptive_style


def cadt_gram(gram, con_sim):
    adaptive_gram = []
    for g in range(len(gram)):
        (b, n1, n2) = gram[g].size()
        fg = gram[g].view(b, n1 * n2)
        adaptive_gram.append(con_sim @ fg)
        adaptive_gram[g] = adaptive_gram[g].view(b, n1, n2)
    return adaptive_gram

