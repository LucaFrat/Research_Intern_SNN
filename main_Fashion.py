import torch 
import constants as c
import numpy as np
import my_utils_Fashion 
import time



def main_Fashion():

    device = my_utils_Fashion.set_seed_and_find_device()    
    train_loader, test_loader = my_utils_Fashion.get_Fashion_Dataloaders()

    st = time.time()

    for i, surr_func in enumerate(c.SPIKE_GRADS):
        net = my_utils_Fashion.Net(surrogate_func=surr_func, dataset=1).to(device)
        
        print(f"Surrogate: {c.SURR_FUNCTIONS[i]}")

        output = my_utils_Fashion.training(net=net,
                                         train_loader=train_loader, 
                                         test_loader=test_loader, 
                                         device=device)
        del net
        
        # Save data into .npy files
        spks = {'s1': output[4], 's2': output[5], 's3': output[6]}
        mems = {'m1': output[8], 'm2': output[9], 'm3': output[10]}
        np.save(f'Outputs/metrics_{c.SURR_FUNCTIONS[i]}.npy', np.array([output[:4]])) 
        np.save(f'Outputs/Spks_{c.SURR_FUNCTIONS[i]}.npy', spks)
        np.save(f'Outputs/Spks_tot_{c.SURR_FUNCTIONS[i]}.npy', np.array(output[7]))
        np.save(f'Outputs/Mems_{c.SURR_FUNCTIONS[i]}.npy', mems)
        np.save(f'Outputs/Mems_tot_{c.SURR_FUNCTIONS[i]}.npy', np.array(output[11]))

    en = time.time()

    print(f'Run time: {(en-st)/60:.2f} min\n') 



if __name__ == '__main__':
    main_Fashion()