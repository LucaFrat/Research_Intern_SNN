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


def get_network(device):
    spike_grad = surrogate.atan()
    params = c.Net()
    #  Initialize Network
    net = nn.Sequential(nn.Conv2d(params.channels[0], params.channels[1], params.kernels[0]),
                        snn.Leaky(beta=c.beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(2),
                        nn.Conv2d(params.channels[1], params.channels[2], params.kernels[1]),
                        snn.Leaky(beta=c.beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(params.channels[-1]*5*5, params.classes),
                        snn.Leaky(beta=c.beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)
    return net


def forward_pass(net, data):
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(data.size(0)):  # data.size(0) = number of time steps
      spk_out, mem_out = net(data[step])
      spk_rec.append(spk_out)

  return torch.stack(spk_rec)


def train_NMIST(trainloader, testloader, net, device):
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    optimizer = torch.optim.Adam(net.parameters(), lr=c.lr, betas=(c.betas_Adam[0], c.betas_Adam[1]))
    loss_fn = SF.mse_count_loss(correct_rate=c.correct_rate, incorrect_rate=1-c.correct_rate)

    count = 1
    # training loop
    for epoch in range(c.epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device)
            targets = targets.to(device)

            net.train()
            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            with torch.no_grad():
                total = 0
                test_acc = 0
                net.eval()
                
                testloader = iter(testloader)
                count1 = 0
                for data, targets in testloader:
                    if count1 > 5:
                        break
                    data = data.to(device)
                    targets = targets.to(device)
                    spk_rec = forward_pass(net, data)
                    test_acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                    total += spk_rec.size(1)
                    count1 += 1
                test_acc_hist.append(test_acc/total)
                print(f"Test Accuracy: {test_acc/total * 100:.2f}%")

            # Store loss history for future plotting
            # train_loss_hist.append(loss_val.item())
            train_acc = SF.accuracy_rate(spk_rec, targets)
            train_acc_hist.append(train_acc)

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
            print(f"Train Accuracy: {train_acc * 100:.2f}%\n")

            count += 1
            # This will end training after 'num_iters' iterations by default
            if i == c.num_iters:
                break

    return train_acc_hist, test_acc_hist 


    

def plot_loss_NMNIST(train_acc, test_acc):
    fig = plt.figure(facecolor="w")
    plt.plot(train_acc, label='Train Acc', linestyle='-')
    plt.plot(test_acc, label='Test Acc', linestyle='--')
    plt.title("Train-Test Set Accuracy")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.show()
