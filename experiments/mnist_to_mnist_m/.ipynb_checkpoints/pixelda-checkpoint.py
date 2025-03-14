import argparse
import os
import numpy as np
import math
import itertools
import sys
sys.path.insert(0, '../../')
sys.path.append('/datacommons/carlsonlab/yl407/packages')
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision.datasets import MNIST
from data.mnist_mnistm_data import *
from torch.nn.utils import spectral_norm
from models.model_mnist_mnistm import *
import utils.config as config
from utils.helper import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm, trange
from utils.helper import *
#os.makedirs("images", exist_ok=True)
seed_new = 123
seed_torch(seed=seed_new)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the noise input")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes in the dataset")
parser.add_argument("--sample_interval", type=int, default=300, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / 2 ** 4)
patch = (1, patch, patch)
cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Fully-connected layer which constructs image channel shaped output from noise
        self.fc = nn.Linear(opt.latent_dim, opt.channels * opt.img_size ** 2)

        self.l1 = nn.Sequential(nn.Conv2d(opt.channels * 2, 64, 3, 1, 1), nn.ReLU(inplace=True))

        resblocks = []
        for _ in range(opt.n_residual_blocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)

        self.l2 = nn.Sequential(nn.Conv2d(64, opt.channels, 3, 1, 1), nn.Tanh())

    def forward(self, img, z):
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)

        return img_


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False), *block(64, 128), *block(128, 256), *block(256, 512)
        )

        input_size = 2 #opt.img_size // 2 ** 4
        self.output_layer = nn.Sequential(nn.Linear(512 * input_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label


# Loss function
adversarial_loss = torch.nn.MSELoss()
task_loss = torch.nn.CrossEntropyLoss()

# Loss weights
lambda_adv = 1
lambda_task = 0.1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
classifier = Classifier()

if cuda:
    generator.cuda()
    discriminator.cuda()
    classifier.cuda()
    adversarial_loss.cuda()
    task_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
classifier.apply(weights_init_normal)

# Configure data loader

source_dataset = load_mnist_LS(source_num=600,alpha=8,train_flag=True)
dataloader_A = torch.utils.data.DataLoader(source_dataset, batch_size=opt.batch_size, drop_last=True,\
                           shuffle=True, num_workers=0, pin_memory=True)
    

##########target data########################
target_dataset_label = load_mnistm_LS(target_num=100,alpha=8,MNISTM=MNISTM,train_flag=True)

dataloader_B = torch.utils.data.DataLoader(target_dataset_label, batch_size=opt.batch_size,drop_last=True,\
                           shuffle=True, num_workers=0, pin_memory=True)

eval_dataset = load_mnistm_LS(target_num=100,alpha=8,MNISTM=MNISTM,train_flag=False)

dataloader_eval = torch.utils.data.DataLoader(eval_dataset, batch_size=len(eval_dataset),drop_last=False,\
                           shuffle=False, num_workers=0, pin_memory=True)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(generator.parameters(), classifier.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

# Keeps 100 accuracy measurements
task_performance = []
target_performance = []
best_target_acc = 0
for epoch in range(opt.n_epochs):
    for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):

        batch_size = imgs_A.size(0)

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, *patch).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs_A = Variable(imgs_A.type(FloatTensor).expand(batch_size, 3, opt.img_size, opt.img_size))
        labels_A = Variable(labels_A.type(LongTensor))
        imgs_B = Variable(imgs_B.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise
        z = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        fake_B = generator(imgs_A, z)

        # Perform task on translated source image
        label_pred = classifier(fake_B)

        # Calculate the task loss
        task_loss_ = (task_loss(label_pred, labels_A) + task_loss(classifier(imgs_A), labels_A)) / 2

        # Loss measures generator's ability to fool the discriminator        
        g_loss = lambda_adv * adversarial_loss(discriminator(fake_B), valid) + lambda_task * task_loss_

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(imgs_B), valid)
        fake_loss = adversarial_loss(discriminator(fake_B.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ---------------------------------------
        #  Evaluate Performance on target domain
        # ---------------------------------------

        # Evaluate performance on translated Domain A
        with torch.no_grad():
            acc = np.mean(np.argmax(label_pred.data.cpu().numpy(), axis=1) == labels_A.data.cpu().numpy())
            task_performance.append(acc)
            if len(task_performance) > 100:
                task_performance.pop(0)

        #print(
        #    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [CLF acc: %3d%% (%5.3f%%)"
        #    % (
        #        epoch,
        #        opt.n_epochs,
        #        i,
        #        len(dataloader_A),
        #        d_loss.item(),
        #        g_loss.item(),
        #        100 * acc,
        #        100 * np.mean(task_performance),
        #    )
        #)
    with torch.no_grad():
        for imgs_eval, labels_eval in tqdm(dataloader_eval, leave=False):
            imgs_eval = Variable(imgs_eval.type(FloatTensor))
            pred_eval = classifier(imgs_eval)
            target_acc = np.mean(np.argmax(pred_eval.data.cpu().numpy(), axis=1) == labels_eval.numpy())
            if target_acc > best_target_acc:
                best_target_acc = target_acc
            print("Target_accuracy",100*target_acc)
            print("best ACC:", 100*best_target_acc)
            