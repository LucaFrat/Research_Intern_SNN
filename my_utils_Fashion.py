import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import tonic
from torch.utils.data import DataLoader
import numpy as np
import random
import constants as c
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
from snntorch import spikegen
import torch.nn as nn
import matplotlib.pyplot as plt


def set_seed_and_find_device():
    random.seed(c.SEED)
    torch.manual_seed(c.SEED)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_Fashion_Dataloaders():
    data_path='./data/Fashion_MNIST'
    transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,))
                    ])
    fashion_train = torchvision.datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

    fashion_train = torch.utils.data.Subset(fashion_train, range(0, len(fashion_train), 20))
    fashion_test = torch.utils.data.Subset(fashion_test, range(0, len(fashion_test), 20))

    train_loader = DataLoader(fashion_train, batch_size=c.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(fashion_test, batch_size=c.BATCH_SIZE, shuffle=True, drop_last=True)

    return train_loader, test_loader



# Define NetworK: Input layer - 2 onv+LIF layers with same dimensions - Output layer
class Net(nn.Module):
    def __init__(self, dataset:int=0):
        super().__init__()

        params = [c.NMNIST_Net(), c.FashionMNIST_Net(), c.DVS_Net()][dataset]                        

        # Initialize layers
        self.conv1 = nn.Conv2d(params.CHANNELS[0], params.CHANNELS[1], params.KERNELS[0])
        self.lif1 = snn.Leaky(beta=c.BETA, spike_grad=c.SPIKE_GRAD)
        self.conv2 = nn.Conv2d(params.CHANNELS[1], params.CHANNELS[2], params.KERNELS[1])
        self.lif2 = snn.Leaky(beta=c.BETA, spike_grad=c.SPIKE_GRAD)
        self.fc1 = nn.Linear(params.CHANNELS[-1]*params.RES_DIM*params.RES_DIM, params.CLASSES)
        self.lif3 = snn.Leaky(beta=c.BETA, spike_grad=c.SPIKE_GRAD)

    def forward(self, x):
        
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()  
        
        # Record the LIF layers
        spk1_rec = []
        spk2_rec = []
        spk_out_rec = []

        for step in range(x.size(0)):
            cur1 = F.max_pool2d(self.conv1(x[step]), 2)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = F.max_pool2d(self.conv2(spk1), 2)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(spk2.view(c.BATCH_SIZE, -1))
            spk_out, mem3 = self.lif3(cur3, mem3)
            
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk_out_rec.append(spk_out)            
                
        return torch.stack(spk1_rec), torch.stack(spk2_rec), torch.stack(spk_out_rec)





def spike_count_neuron_wise(spk1, spk2, spk_out):
    with torch.no_grad():
        # tot number of spikes per neuron
        # sum over the 3 dimensions: num_steps, batch_size, channels 
        tot_spk1 = spk1.mean(dim=0).mean(dim=0).sum(dim=0)
        tot_spk2 = spk2.mean(dim=0).mean(dim=0).sum(dim=0)
        tot_spk_out = spk_out.mean(dim=0).mean(dim=0)
    return tot_spk1, tot_spk2, tot_spk_out

def spike_count_layer_wise(spk1_nw, spk2_nw, spk_out_nw):
    with torch.no_grad():
        # average probability of spike per each layer 
        # (not neuron but layer wise)
        tot_spk1 = spk1_nw.mean(dim=0).mean(dim=0)
        tot_spk2 = spk2_nw.mean(dim=0).mean(dim=0)
        tot_spk_out = spk_out_nw.mean(dim=0)
    return tot_spk1, tot_spk2, tot_spk_out





def training(net, train_loader, test_loader, device):
    
    loss_hist = []
    loss_epoch_hist=[]
    acc_hist = []
    acc_epoch_hist=[]
    
    test_loss_hist = []
    test_loss_epoch_hist=[]
    test_acc_hist = []
    test_acc_epoch_hist=[]

    spk_layer1 = []
    spk_layer2 = []
    spk_layer_out = []
    spk_tot_epochs = []
    
    #loss and optimizer
    loss_fn = SF.mse_count_loss(correct_rate=c.CORRECT_RATE, incorrect_rate=1-c.CORRECT_RATE)
    optimizer = torch.optim.Adam(net.parameters(), lr=c.LR, betas=(c.BETAS_ADAM[0], c.BETAS_ADAM[1]))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Outer training loop
    for epoch in range(c.EPOCHS):

        print(f"\nEpoch: {epoch}")

        for data, targets in iter(train_loader):
            data = spikegen.rate(data, num_steps=c.NUM_STEPS)
            data = data.to(device)
            targets = targets.to(device)
           
            # forward pass
            net.train()
            spk1, spk2, spk_out = net(data)

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=c.DTYPE, device=device)
            for _ in range(c.NUM_STEPS):
                loss_val += loss_fn(spk_out, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            # scheduler.step()
          
            # Store loss and accuracy history for future plotting
            loss_hist.append(loss_val.item())
            acc = SF.accuracy_rate(spk_out, targets)
            acc_hist.append(acc)
            
            #Test set   
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = spikegen.rate(test_data, num_steps=c.NUM_STEPS)
                test_targets = test_targets.to(device)

                # Test set forward pass
                spk1_test, spk2_test, spk_out_test = net(test_data)

                # Test set loss 
                test_loss = torch.zeros((1), dtype=c.DTYPE, device=device)
                for _ in range(c.NUM_STEPS):
                    test_loss += loss_fn(spk_out_test, test_targets) 

                test_loss_hist.append(test_loss.item())

                #test set accuracy
                test_acc = SF.accuracy_rate(spk_out_test, test_targets)
                test_acc_hist.append(test_acc)

	
        loss_epoch = np.mean(loss_hist)
        acc_epoch = np.mean(acc_hist)
        test_loss_epoch = np.mean(test_loss_hist)
        test_acc_epoch = np.mean(test_acc_hist)
        
        loss_epoch_hist.append(loss_epoch)
        acc_epoch_hist.append(acc_epoch)
        test_loss_epoch_hist.append(test_loss_epoch)
        test_acc_epoch_hist.append(test_acc_epoch)

        del acc_epoch
        del loss_epoch
        del test_loss_epoch
        del test_acc_epoch

        
        #compute the total number of spikes for each neuron and layer
        spk1_tot_nw, spk2_tot_nw, spk_out_tot_nw = spike_count_neuron_wise(spk1, spk2, spk_out)
        spk1_tot_lw, spk2_tot_lw, spk_out_tot_lw = spike_count_layer_wise(spk1_tot_nw, spk2_tot_nw, spk_out_tot_nw)

        #Store the total number of spikes for each layer
        spk_layer1.append(spk1_tot_nw.cpu())
        spk_layer2.append(spk2_tot_nw.cpu())
        spk_layer_out.append(spk_out_tot_nw.cpu())
        spk_tot_epochs.append(np.array([spk1_tot_lw, spk2_tot_lw, spk_out_tot_lw]))
 
    return loss_epoch_hist, test_loss_epoch_hist, acc_epoch_hist, test_acc_epoch_hist,\
            spk_layer1, spk_layer2, spk_layer_out, np.array(spk_tot_epochs)
