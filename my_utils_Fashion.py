import torch 
import torchvision
import torchvision.transforms as transforms
import tonic
from torch.utils.data import DataLoader
import numpy as np
import constants as c
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import torch.nn as nn
import matplotlib.pyplot as plt




def get_Fashion_Dataloaders():
    data_path='./data/mnist'
    transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,))
                    ])
    train = torchvision.datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    test = torchvision.datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=c.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=c.BATCH_SIZE, shuffle=True, drop_last=True)

    return train_loader, test_loader


def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
    return torch.stack(spk_rec), torch.stack(mem_rec)


def batch_accuracy(train_loader, net, device):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, c.NUM_STEP, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total


def train(train_loader, test_loader, net, device):
    optimizer = torch.optim.Adam(net.parameters(), lr=c.LR, betas=(c.BETAS_ADAM[0], c.BETAS_ADAM[1]))
    loss_fn = SF.ce_rate_loss()

    loss_hist = []
    test_acc_hist = []
    counter = 0

    # Outer training loop
    for epoch in range(c.EPOCHS):

        # Training loop
        for data, targets in iter(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, _ = forward_pass(net, c.NUM_STEP, data)

            # initialize the loss & sum over time
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            if counter % 50 == 0:
                with torch.no_grad():
                    net.eval()

                    # Test set forward pass
                    test_acc = batch_accuracy(test_loader, net, device)
                    print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                    test_acc_hist.append(test_acc.item())

            counter += 1
