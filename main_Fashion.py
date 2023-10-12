import torch 
import constants as c
import my_utils
import numpy as np
import my_utils_Fashion 
import time
import random


def main_Fashion():

    random.seed(c.SEED)
    torch.manual_seed(c.SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = my_utils_Fashion.get_Fashion_Dataloaders()
    net = my_utils_Fashion.Net(dataset=1).to(device)


    st = time.time()
    loss_hist, test_loss_hist, acc_hist, test_acc_hist, \
    layer1, layer2, layer_out  = my_utils_Fashion.training(net=net,
                                                            train_loader=train_loader, 
                                                            test_loader=test_loader, 
                                                            device=device)
    en = time.time()

    #return a file with x=epochs, y_1line=loss, y_2line=test_loss, y_3line=acc, y_4line=test acc
    # np.save('3h_stat_nowd.npy', np.vstack((loss_hist, test_loss_hist, acc_hist, test_acc_hist))) 
    # np.save('3h_spk_nowd.npy', np.vstack((layer1,layer2, layer_out)))

    # my_utils.plot_loss(train_acc=train_acc, test_acc=test_acc)

    print(f'Execution time: {(en-st)/60:.2f} min') 




if __name__=="__main__":
    main_Fashion()