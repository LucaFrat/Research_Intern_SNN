import torch 
import torchvision
import torchvision.transforms as Ptransforms
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
import constants as c
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import torch.nn as nn
import matplotlib.pyplot as plt



def get_Fashion_Dataloaders():
    trainset = tonic.datasets.NMNIST(save_to='./data/Fashion', train=True)
    testset = tonic.datasets.NMNIST(save_to='./data/Fashion', train=False)

    transform = tonic.transforms.Compose([torch.from_numpy,
                                        torchvision.transforms.RandomRotation([-c.ROTATION,c.ROTATION])])
    cached_trainset = DiskCachedDataset(trainset, 
                                        transform=transform, 
                                        cache_path='./cache/fashion/train')
    # no augmentations for the testset
    cached_testset = DiskCachedDataset(testset, 
                                       cache_path='./cache/fashion/test')
    trainloader = DataLoader(cached_trainset, 
                             batch_size=c.BATCH_SIZE, 
                             shuffle=True)
    testloader = DataLoader(cached_testset, 
                            batch_size=c.BATCH_SIZE)

    return trainloader, testloader



