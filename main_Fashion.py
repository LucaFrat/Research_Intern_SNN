import torch 
import constants as c
import my_utils
import my_utils_Fashion 
import time
import random


def main_Fashion():
    random.seed(c.SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = my_utils_Fashion.get_Fashion_Dataloaders()

# FROM HERE
    event_tensor = next(iter(trainloader))
    print(event_tensor.shape)

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
    main_Fashion()