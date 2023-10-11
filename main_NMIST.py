import torch 
import constants as c
import my_utils as my_utils
import time
import random


def main_NMIST():
    random.seed(20)
    torch.manual_seed(c.SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = my_utils.get_NMIST_Dataloaders()

    net = my_utils.get_network(device=device)

    st = time.time()
    train_acc, test_acc = my_utils.train(trainloader=trainloader, 
                                               testloader=testloader,
                                               net=net,
                                               device=device)
    en = time.time()

    my_utils.plot_loss(train_acc=train_acc, test_acc=test_acc)

    print(f'Execution time: {(en-st)/60:.2f} min') 


if __name__=="__main__":
    main_NMIST()
