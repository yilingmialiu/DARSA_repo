import sys
sys.path.append('/datacommons/carlsonlab/yl407/packages')
sys.path.insert(0, '../../')

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import utils.config as config
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from utils.helper import *

class BSDS500(Dataset):

    def __init__(self):
        image_folder = config.DATA_DIR / 'BSDS500/data/images'
        self.image_files = list(map(str, image_folder.glob('*/*.jpg')))

    def __getitem__(self, i):
        image = cv2.imread(self.image_files[i], cv2.IMREAD_COLOR)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor

    def __len__(self):
        return len(self.image_files)


class MNISTM(Dataset):
    def __init__(self,mnist_label_shift):
        super(MNISTM, self).__init__()
        self.mnist = mnist_label_shift
        self.bsds = BSDS500()
        # Fix RNG so the same images are used for blending
        self.rng = np.random.RandomState(42)

    def __getitem__(self, i):
        digit, label = self.mnist[i]
        digit = transforms.ToTensor()(digit)
        bsds_image = self._random_bsds_image()
        patch = self._random_patch(bsds_image)
        patch = patch.float() / 255
        blend = torch.abs(patch - digit)
        return blend, label

    def _random_patch(self, image, size=(28, 28)):
        _, im_height, im_width = image.shape
        x = self.rng.randint(0, im_width-size[1])
        y = self.rng.randint(0, im_height-size[0])
        return image[:, y:y+size[0], x:x+size[1]]

    def _random_bsds_image(self):
        i = self.rng.choice(len(self.bsds))
        return self.bsds[i]

    def __len__(self):
        return len(self.mnist)




class MNISTM_transform(Dataset):
    def __init__(self, mnist_label_shift, transform=None):
        super(MNISTM_transform, self).__init__()
        self.mnist = mnist_label_shift
        self.bsds = BSDS500()
        self.rng = np.random.RandomState(42)
        self.transform = transform

    def __getitem__(self, i):
        digit, label = self.mnist[i]
        digit_tensor = transforms.ToTensor()(digit).float()

        bsds_image = self._random_bsds_image()
        patch = self._random_patch(bsds_image)

        # Ensure digit has 3 channels to match the BSDS patch
        if digit_tensor.shape[0] == 1:
            digit_tensor = digit_tensor.repeat(3, 1, 1)

        # Resize digit if necessary to match patch size (should be 28x28)
        if digit_tensor.shape[1:] != patch.shape[1:]:
            digit_tensor = transforms.Resize(patch.shape[1:])(digit_tensor)

        blend = torch.abs(patch - digit_tensor)

        if self.transform:
            # Convert blend tensor to PIL Image
            blend_pil = transforms.ToPILImage()(blend)
            # Apply the transformations
            blend = self.transform(blend_pil)

        return blend, label

    def _random_patch(self, image, size=(28, 28)):
        c, im_height, im_width = image.shape
        x = self.rng.randint(0, im_width - size[1])
        y = self.rng.randint(0, im_height - size[0])
        return image[:, y:y + size[0], x:x + size[1]]

    def _random_bsds_image(self):
        return self.bsds[self.rng.choice(len(self.bsds))]

    def __len__(self):
        return len(self.mnist)
    

class MNISTM_DRANet(Dataset):

    def __init__(self,mnist_label_shift):
        super(MNISTM_DRANet, self).__init__()
        self.mnist = mnist_label_shift
        self.bsds = BSDS500()
        # Fix RNG so the same images are used for blending
        self.rng = np.random.RandomState(42)

    def __getitem__(self, i):
        digit, label = self.mnist[i]
        digit = transforms.ToTensor()(digit)
        bsds_image = self._random_bsds_image()
        patch = self._random_patch(bsds_image)
        patch = patch.float() / 255
        blend = torch.abs(patch - digit)
        return blend, label

    def _random_patch(self, image, size=(64, 64)):
        _, im_height, im_width = image.shape
        x = self.rng.randint(0, im_width-size[1])
        y = self.rng.randint(0, im_height-size[0])
        return image[:, y:y+size[0], x:x+size[1]]

    def _random_bsds_image(self):
        i = self.rng.choice(len(self.bsds))
        return self.bsds[i]

    def __len__(self):
        return len(self.mnist)
    

def load_mnist_LS(source_num,alpha,train_flag=True):
    source_dataset = MNIST('../../data/mnist', train=train_flag, download=True,
                          transform=Compose([GrayscaleToRgb(), ToTensor()]))
    indices_source = []
    
    for i in range(10):
        if i%2 == 0:
            indices_even_temp = np.where((source_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_even_temp,source_num,replace=False)
        if i%2 != 0:
            indices_odd_temp = np.where((source_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_odd_temp,source_num*alpha,replace=False)
        indices_source = np.concatenate((indices_source, indices_temp_sample))
        np.random.shuffle(indices_source)

    source_dataset.data = source_dataset.data[indices_source]
    source_dataset.targets = source_dataset.targets[indices_source]
    return source_dataset






def load_mnist(train_flag=True):
    source_dataset = MNIST('../../data/mnist', train=train_flag, download=True,
                          transform=Compose([GrayscaleToRgb(), ToTensor()]))
    return source_dataset


def load_mnist_dominate(source_num,digit,train_flag=True):
    source_dataset = MNIST('../../data/mnist', train=train_flag, download=True,
                          transform=Compose([GrayscaleToRgb(), ToTensor()]))
    indices_source = []
    
    for i in range(10):
        if i == digit:
            indices_even_temp = np.where((source_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_even_temp,source_num,replace=False)
        if i!= digit:
            indices_odd_temp = np.where((source_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_odd_temp,int(source_num/40),replace=False)
        indices_source = np.concatenate((indices_source, indices_temp_sample))
        np.random.shuffle(indices_source)

    source_dataset.data = source_dataset.data[indices_source]
    source_dataset.targets = source_dataset.targets[indices_source]
    return source_dataset





def load_mnist_LS_DRANet(source_num,alpha,train_flag=True):    
    
    source_dataset = MNIST('../../data/mnist', train=train_flag,
                              download=True,\
                              transform=transforms.Compose([
                                       transforms.Resize(64),
                                       GrayscaleToRgb(),
                                       transforms.ToTensor()]))
    
    indices_source = []
    
    for i in range(10):
        if i%2 == 0:
            indices_even_temp = np.where((source_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_even_temp,source_num,replace=False)
        if i%2 != 0:
            indices_odd_temp = np.where((source_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_odd_temp,source_num*alpha,replace=False)
        indices_source = np.concatenate((indices_source, indices_temp_sample))
        np.random.shuffle(indices_source)

    source_dataset.data = source_dataset.data[indices_source]
    source_dataset.targets = source_dataset.targets[indices_source]
    return source_dataset

def load_mnist_DRANet(train_flag=True):    
    
    source_dataset = MNIST('../../data/mnist', train=train_flag,
                              download=True,\
                              transform=transforms.Compose([
                                       transforms.Resize(64),
                                       GrayscaleToRgb(),
                                       transforms.ToTensor()]))
    return source_dataset


def load_mnistm(MNISTM,train_flag=False):
    target_dataset = datasets.MNIST('../../data/mnist',train=train_flag,download=True)
    target_dataset_label = MNISTM(mnist_label_shift=target_dataset)
    return target_dataset_label



def load_mnistm_LS(target_num,MNISTM,alpha,train_flag=False):
    
    target_dataset = datasets.MNIST('../../data/mnist',train=train_flag,download=True)
    indices_target = []
    
    for i in range(10):
        if i%2 == 0:
            indices_even_temp = np.where((target_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_even_temp,target_num*alpha,replace=False)
        if i%2 != 0:
            indices_odd_temp = np.where((target_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_odd_temp,target_num,replace=False)
        indices_target = np.concatenate((indices_target, indices_temp_sample))
        np.random.shuffle(indices_target)

    target_dataset.data = target_dataset.data[indices_target]
    target_dataset.targets = target_dataset.targets[indices_target]

    target_dataset_label = MNISTM(mnist_label_shift=target_dataset)
    
    return target_dataset_label


def load_mnistm_LS_transform(target_num, MNISTM, alpha, train_flag=False, transform=None):

    target_dataset = datasets.MNIST('../../data/mnist', train=train_flag, download=True)
    indices_target = []

    for i in range(10):
        if i % 2 == 0:
            indices_even_temp = np.where((target_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_even_temp, target_num * alpha, replace=False)
        if i % 2 != 0:
            indices_odd_temp = np.where((target_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_odd_temp, target_num, replace=False)
        indices_target = np.concatenate((indices_target, indices_temp_sample))
        np.random.shuffle(indices_target)

    target_dataset.data = target_dataset.data[indices_target]
    target_dataset.targets = target_dataset.targets[indices_target]

    target_dataset_label = MNISTM(mnist_label_shift=target_dataset, transform=transform)  # Pass the transform

    return target_dataset_label


def load_mnistm_dominate(target_num,MNISTM,digit,train_flag=False):
    
    target_dataset = datasets.MNIST('../../data/mnist',train=train_flag,download=True)
    indices_target = []
    
    for i in range(10):
        if i == digit:
            indices_even_temp = np.where((target_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_even_temp,target_num,replace=False)
        if i != digit:
            indices_odd_temp = np.where((target_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_odd_temp,int(target_num/40),replace=False)
        indices_target = np.concatenate((indices_target, indices_temp_sample))
        np.random.shuffle(indices_target)

    target_dataset.data = target_dataset.data[indices_target]
    target_dataset.targets = target_dataset.targets[indices_target]

    target_dataset_label = MNISTM(mnist_label_shift=target_dataset)
    
    return target_dataset_label


def load_mnistm_LS_DRANet(target_num,MNISTM,alpha,train_flag=False):
    
    target_dataset = datasets.MNIST('../../data/mnist',train=train_flag,download=True,\
                                      transform=transforms.Compose([transforms.Resize(64)]))
    indices_target = []
    
    for i in range(10):
        if i%2 == 0:
            indices_even_temp = np.where((target_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_even_temp,target_num*alpha,replace=False)
        if i%2 != 0:
            indices_odd_temp = np.where((target_dataset.targets == i))[0]
            indices_temp_sample = np.random.choice(indices_odd_temp,target_num,replace=False)
        indices_target = np.concatenate((indices_target, indices_temp_sample))
        np.random.shuffle(indices_target)

    target_dataset.data = target_dataset.data[indices_target]
    target_dataset.targets = target_dataset.targets[indices_target]

    target_dataset_label = MNISTM(mnist_label_shift=target_dataset)
    
    return target_dataset_label