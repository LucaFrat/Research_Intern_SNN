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

    fashion_train = torch.utils.data.Subset(fashion_train, range(0, len(fashion_train), c.SUBSET))
    fashion_test = torch.utils.data.Subset(fashion_test, range(0, len(fashion_test), c.SUBSET))

    train_loader = DataLoader(fashion_train, batch_size=c.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(fashion_test, batch_size=c.BATCH_SIZE, shuffle=True, drop_last=True)

    return train_loader, test_loader



# Define NetworK: Input layer - 2 Conv+LIF layers - Output layer
class Net(nn.Module):
    def __init__(self, surr_info):
        super().__init__()

        params = c.FashionMNIST_Net()
        surr_func = c.get_surrogate_function(*surr_info)

        # Initialize layers
        self.conv1 = nn.Conv2d(params.CHANNELS[0], params.CHANNELS[1], params.KERNELS[0])
        self.lif1 = snn.Leaky(beta=c.BETA, spike_grad=surr_func)
        self.conv2 = nn.Conv2d(params.CHANNELS[1], params.CHANNELS[2], params.KERNELS[1])
        self.lif2 = snn.Leaky(beta=c.BETA, spike_grad=surr_func)
        self.fc1 = nn.Linear(params.CHANNELS[-1]*params.RES_DIM*params.RES_DIM, params.CLASSES)
        self.lif3 = snn.Leaky(beta=c.BETA, spike_grad=surr_func)

    def forward(self, x):
        
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()  
        
        # Record the LIF layers
        spk1_rec = []
        spk2_rec = []
        spk_out_rec = []

        mem1_rec = []
        mem2_rec = []
        mem_out_rec = []

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

            mem1_rec.append(mem1)
            mem2_rec.append(mem2)
            mem_out_rec.append(mem3)          
                
        return torch.stack(spk1_rec), torch.stack(spk2_rec), torch.stack(spk_out_rec),\
                torch.stack(mem1_rec), torch.stack(mem2_rec), torch.stack(mem_out_rec)





def count_neuron_wise(in1, in2, in3):
    with torch.no_grad():
        # tot number of Spikes/Membrane_potential per neuron
        # sum over the 3 dimensions: num_steps, batch_size, channels 
        out1 = in1.mean(dim=0).mean(dim=0).mean(dim=0)
        out2 = in2.mean(dim=0).mean(dim=0).mean(dim=0)
        out3 = in3.mean(dim=0).mean(dim=0)
    return out1, out2, out3

def count_layer_wise(in1, in2, in3):
    with torch.no_grad():
        # average probability of Spike/Membrane_potential per layer 
        # (not neuron but layer wise)
        out1 = in1.mean(dim=0).mean(dim=0)
        out2 = in2.mean(dim=0).mean(dim=0)
        out3 = in3.mean(dim=0)
    return out1, out2, out3





def training(net, train_loader, test_loader, device):
    
    train_acc_hist = []
    train_acc_epoch_hist=[]
    
    test_acc_hist = []
    test_acc_epoch_hist=[]

    spk_layer1 = []
    spk_layer2 = []
    spk_layer_out = []
    spk_tot_epochs = []

    mem_layer1 = []
    mem_layer2 = []
    mem_layer_out = []
    mem_tot_epochs = []
    
    #loss and optimizer
    loss_fn = SF.mse_count_loss(correct_rate=c.CORRECT_RATE, incorrect_rate=1-c.CORRECT_RATE)
    optimizer = torch.optim.Adam(net.parameters(), lr=c.LR, betas=(c.BETAS_ADAM[0], c.BETAS_ADAM[1]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Outer training loop
    for epoch in range(c.EPOCHS):

        if epoch%50 == 0:
            print(f"Epoch: {epoch}")

        for data, targets in iter(train_loader):
            data = spikegen.rate(data, num_steps=c.NUM_STEPS)
            data = data.to(device)
            targets = targets.to(device)
           
            # forward pass
            net.train()
            spk1, spk2, spk_out, mem1, mem2, mem_out = net(data)

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=c.DTYPE, device=device)
            for _ in range(c.NUM_STEPS):
                loss_val += loss_fn(spk_out, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            scheduler.step()
          
            # Store loss and accuracy history for future plotting
            train_acc = SF.accuracy_rate(spk_out, targets)
            train_acc_hist.append(train_acc)
            
            #Test set   
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = spikegen.rate(test_data, num_steps=c.NUM_STEPS)
                test_targets = test_targets.to(device)

                # Test set forward pass
                _, _, spk_out_test, _, _, _ = net(test_data)

                # Test set loss 
                test_loss = torch.zeros((1), dtype=c.DTYPE, device=device)
                for _ in range(c.NUM_STEPS):
                    test_loss += loss_fn(spk_out_test, test_targets) 

                #test set accuracy
                test_acc = SF.accuracy_rate(spk_out_test, test_targets)
                test_acc_hist.append(test_acc)

	
        train_acc_epoch = np.mean(train_acc_hist)
        test_acc_epoch = np.mean(test_acc_hist)
        
        train_acc_epoch_hist.append(train_acc_epoch)
        test_acc_epoch_hist.append(test_acc_epoch)

        del train_acc_epoch
        del test_acc_epoch

        
        # Compute the total number of spikes neuron and layer wise
        spk1_tot_nw, spk2_tot_nw, spk_out_tot_nw = count_neuron_wise(spk1, spk2, spk_out)
        spk1_tot_lw, spk2_tot_lw, spk_out_tot_lw = count_layer_wise(spk1_tot_nw, spk2_tot_nw, spk_out_tot_nw)

        # Compute the average membrane potential neuron and layer wise
        mem1_tot_nw, mem2_tot_nw, mem_out_tot_nw = count_neuron_wise(mem1, mem2, mem_out)
        mem1_tot_lw, mem2_tot_lw, mem_out_tot_lw = count_layer_wise(mem1_tot_nw, mem2_tot_nw, mem_out_tot_nw)

        # Store the total number of spikes for each layer
        spk_layer1.append(spk1_tot_nw)
        spk_layer2.append(spk2_tot_nw)
        spk_layer_out.append(spk_out_tot_nw)
        spk_tot_epochs.append(np.array([spk1_tot_lw, spk2_tot_lw, spk_out_tot_lw]))

        mem_layer1.append(mem1_tot_nw)
        mem_layer2.append(mem2_tot_nw)
        mem_layer_out.append(mem_out_tot_nw)
        mem_tot_epochs.append(np.array([mem1_tot_lw, mem2_tot_lw, mem_out_tot_lw]))
 
    return train_acc_epoch_hist, test_acc_epoch_hist,\
            spk_layer1, spk_layer2, spk_layer_out, spk_tot_epochs, \
            mem_layer1, mem_layer2, mem_layer_out, mem_tot_epochs
