import torch 
import constants as c
import numpy as np
import my_utils_Fashion 
import time



def main_Fashion():

    device = my_utils_Fashion.set_seed_and_find_device()    
    train_loader, test_loader = my_utils_Fashion.get_Fashion_Dataloaders()

    sequence = ['First', 'Second']

    st = time.time()

    for i, surr_func in enumerate(c.SURR_FUNCTIONS):
        net = my_utils_Fashion.Net(surr_func=surr_func, dataset=1).to(device)
        
        print(f"\nSurrogate: {c.SURR_NAMES[i]}\n")

        output = my_utils_Fashion.training(net=net,
                                            train_loader=train_loader, 
                                            test_loader=test_loader, 
                                            device=device)
        del net
        
        # Save data into .npy files
        np.save(f'Outs_Check_Training_Order/Acc_{c.SURR_NAMES[i]}_{sequence[i]}.npy', np.array([output[2:4]])) 
        np.save(f'Outs_Check_Training_Order/Spks_tot_{c.SURR_NAMES[i]}_{sequence[i]}.npy', np.array(output[7]))
        
        # spks = {'s1': output[4], 's2': output[5], 's3': output[6]}
        # mems = {'m1': output[8], 'm2': output[9], 'm3': output[10]}
        # np.save(f'Outs_Check_Training_Order/Spks_{c.SURR_NAMES[s_now]}.npy', spks)
        # np.save(f'Outs_Check_Training_Order/Mems_{c.SURR_NAMES[s_now]}.npy', mems)
        # np.save(f'Outs_Check_Training_Order/Mems_tot_{c.SURR_NAMES[s_now]}.npy', np.array(output[11]))

    en = time.time()

    print(f'Run time: {(en-st)/60:.2f} min\n') 



if __name__ == '__main__':
    main_Fashion()