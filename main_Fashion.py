import torch 
import constants as c
import numpy as np
import my_utils_Fashion 
import time



def main_Fashion():

    device = my_utils_Fashion.set_seed_and_find_device()    
    train_loader, test_loader = my_utils_Fashion.get_Fashion_Dataloaders()


    st = time.time()

    for surr_func in c.SPIKE_GRAD:
        net = my_utils_Fashion.Net(surrogate_func=surr_func, dataset=1).to(device)
        output = my_utils_Fashion.training(net=net,
                                         train_loader=train_loader, 
                                         test_loader=test_loader, 
                                         device=device)
        del net
        print(f"Accuracy: {output[2]}")
        print(f"Tot prob of spike in network: {np.sum(output[7], axis=1)}\n")

    en = time.time()
    print(f'Execution time: {(en-st)/60:.2f} min\n') 




if __name__ == '__main__':
    main_Fashion()