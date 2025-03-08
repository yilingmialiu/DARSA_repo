from PIL import Image
import numpy as np
import torch
from torch.autograd import Function
from sklearn.cluster import KMeans
import torch.nn.functional as F
import random
import os
from torch.autograd import grad
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loop_iterable(iterable):
    while True:
        yield from iterable
        
        
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
        
class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)
    

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output, None
    
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    
    
###define functions###
def get_cluster(x,K):
    """Get clusters
    #Input:
    ##x: latent representations
    ##K: number of clusters
    #Output:
    ##one hot encoding of clustering ids
    """
    kmeans = KMeans(n_clusters=K, random_state=0).fit(x.cpu())
    return F.one_hot(torch.from_numpy(kmeans.labels_).to(torch.int64)).float().to(device)


def weighted_wass_source(cluster_s,w_imp,h_s,K,critic):
    """Calculate weighted wass distance for source domain
    #Input:
    ##cluster_s: learnt source clustering 
    ##w_imp: importance weighting
    ##h_s: source latent representations
    ##K: number of clusters
    ##critic: domain critic
    #Output:
    ##weighted wass distance for source domain
    """
    weight_source = 0
    for i in range(K):
        weight_source += (cluster_s[:,i]*w_imp[i]*critic(h_s)[:,i]).mean()
    return weight_source

def weighted_wass_target(cluster_t,h_t,K,critic):
    """Calculate weighted wass distance for target domain
    #Input:
    ##cluster_t: learnt target clustering 
    ##h_t: target latent representations
    ##K: number of clusters
    ##critic: domain critic
    #Output:
    ##weighted wass distance for target domain
    """
    weight_target = 0
    for i in range(K):
        weight_target += (cluster_t[:,i]*critic(h_t)[:,i]).mean()
    return weight_target


def gradient_penalty(critic, h_s, h_t):
    """Calculate gradient penalty
    #based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    #Input:
    ##critic: domain critic
    ##h_s: source latent representations
    ##h_t: target latent representations
    #Output:
    ##gradient_penalty
    """
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    
def make_ones(size):
    data = Variable(torch.ones(size, 1))
    return data.to(device)

def make_zeros(size):
    data = Variable(torch.zeros(size, 1))
    return data.to(device)

def sntg_loss_func(cluster,feature,LAMBDA):
    #L_orthogonal
    graph = torch.sum(cluster[:, None, :] * cluster[None, :, :], 2)
    distance = torch.mean((feature[:, None, :] - feature[None, :, :])**2, 2)
    sntg_loss = torch.mean(graph * distance + (1-graph)*torch.nn.functional.relu(LAMBDA- distance))
    return sntg_loss


def centroid_loss_func(K,device,source_y,target_y,source_feature,target_feature):
    ##calculate centroids
    current_source_centroid = torch.zeros(K,source_feature.shape[1],device=device)
    current_target_centroid = torch.zeros(K,target_feature.shape[1],device=device)
    current_source_count = torch.zeros(K,device=device)
    current_target_count = torch.zeros(K,device=device)
    for i in range(K):
        current_source_count[i] = torch.sum(source_y==i)
        current_target_count[i] = torch.sum(target_y==i)
        if torch.sum(source_y==i) > 0 and torch.sum(target_y==i) > 0:
            current_source_centroid[i,:] = source_feature[source_y==i,:].mean(dim=0)
            current_target_centroid[i,:] = target_feature[target_y==i,:].mean(dim=0)

    fm_mask = torch.greater(current_source_count * current_target_count, 0).type(torch.float)
    fm_mask /= torch.mean(fm_mask+1e-8)
    #only use non-zero cluster for calculating centroid loss
    current_source_centroid = current_source_centroid[torch.greater(current_source_count * current_target_count, 0)]
    current_target_centroid = current_target_centroid[torch.greater(current_source_count * current_target_count, 0)]
    fm_mask = fm_mask[torch.greater(current_source_count * current_target_count, 0)]
    centroid_loss = torch.mean(torch.mean(torch.square(current_source_centroid-current_target_centroid),1)*fm_mask)
    return centroid_loss


def save_parameters(mean_clf_loss_all,mean_unweighted_clf_loss_all,mean_centroid_loss_all,\
                    mean_sntg_loss_all,mean_accuracy_all, mean_w1_loss_all,mean_w1_original_all,\
                    key_words):
    torch.save(mean_clf_loss_all,'trained_models/wdgrl_cluster_mean_clf_loss_all_'+key_words+'.pt')
    torch.save(mean_unweighted_clf_loss_all,'trained_models/wdgrl_cluster_mean_unweighted_clf_loss_all_'+key_words+'.pt')
    torch.save(mean_centroid_loss_all,'trained_models/wdgrl_cluster_mean_centroid_loss_all_'+key_words+'.pt')
    torch.save(mean_sntg_loss_all,'trained_models/wdgrl_cluster_mean_sntg_loss_all_'+key_words+'.pt')
    torch.save(mean_w1_loss_all,'trained_models/wdgrl_cluster_mean_w1_loss_all_'+key_words+'.pt')
    torch.save(mean_w1_original_all,'trained_models/wdgrl_cluster_mean_w1_loss_original_all_'+key_words+'.pt')
    torch.save(mean_accuracy_all,'trained_models/wdgrl_cluster_mean_accuracy_all_'+key_words+'.pt')
    