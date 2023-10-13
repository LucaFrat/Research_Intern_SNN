import torch 
import constants as c
import numpy as np
import my_utils_Fashion 
import time


def main_Fashion():

    device = my_utils_Fashion.set_seed_and_find_device()    

    train_loader, test_loader = my_utils_Fashion.get_Fashion_Dataloaders()
    net = my_utils_Fashion.Net(dataset=1).to(device)


    st = time.time()
    loss_hist, test_loss_hist, acc_hist, test_acc_hist, \
    layer1, layer2, layer_out, spk_layers  = my_utils_Fashion.training(net=net,
                                                            train_loader=train_loader, 
                                                            test_loader=test_loader, 
                                                            device=device)
    en = time.time()
    print(f"\nTot spikes per layer per epoch:\n {spk_layers}\n")
    print(f"Accuracy:\n {acc_hist}\n")
    print(f'Execution time: {(en-st)/60:.2f} min\n') 

    #return a file with x=epochs, y_1line=loss, y_2line=test_loss, y_3line=acc, y_4line=test acc
    # np.save('Fashion_test_metrics.npy', np.vstack((loss_hist, test_loss_hist, acc_hist, test_acc_hist))) 
    # np.save('Fashion_test_spks.npy', np.vstack((layer1, layer2, layer_out)))
    # np.save('Fashion_spks_prob_per_layer_per_epoch.npy', spk_layers)




if __name__=="__main__":
    main_Fashion()