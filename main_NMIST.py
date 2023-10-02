import torch 
import constants as c
import utils as my_utils

def main_NMIST():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = my_utils.get_NMIST_Dataloaders()

    event_tensor, target = next(iter(trainloader))
    print(event_tensor.shape)

if __name__=="__main__":
    main_NMIST()
