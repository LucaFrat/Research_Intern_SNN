import torch 
import torchvision
import torchvision.transforms as Ptransforms
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
import constants as c


def get_NMIST_Dataloaders():
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=c.filter_time),
                                      transforms.ToFrame(sensor_size=c.sensor_size,
                                                         time_window=c.time_window)
                                     ])
    trainset = tonic.datasets.NMNIST(save_to='./data', 
                                     transform=frame_transform, 
                                     train=True)
    testset = tonic.datasets.NMNIST(save_to='./data', 
                                    transform=frame_transform, 
                                    train=False)
    transform = tonic.transforms.Compose([torch.from_numpy,
                                        torchvision.transforms.RandomRotation([-c.rRotation,c.rRotation])])
    cached_trainset = DiskCachedDataset(trainset, 
                                        transform=transform, 
                                        cache_path='./cache/nmnist/train')
    # no augmentations for the testset
    cached_testset = DiskCachedDataset(testset, 
                                       cache_path='./cache/nmnist/test')
    trainloader = DataLoader(cached_trainset, 
                             batch_size=c.batch_size, 
                             collate_fn=tonic.collation.PadTensors(batch_first=False), 
                             shuffle=True)
    testloader = DataLoader(cached_testset, 
                            batch_size=c.batch_size, 
                            collate_fn=tonic.collation.PadTensors(batch_first=False))

    return trainloader, testloader