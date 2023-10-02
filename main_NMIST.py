import torch 
import constants as c
import my_utils as my_utils
import time

def main_NMIST():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = my_utils.get_NMIST_Dataloaders()

    event_tensor, target = next(iter(trainloader))
    print(event_tensor.shape)

    net = my_utils.get_network(device=device)

    st = time.time()
    accuracy = my_utils.train_NMIST(trainloader=trainloader, 
                                    net=net,
                                    device=device)
    en = time.time()
    my_utils.plot_loss_NMNIST(acc_hist=accuracy)

    print(f'Execution time: {en-st}s')


if __name__=="__main__":
    main_NMIST()
