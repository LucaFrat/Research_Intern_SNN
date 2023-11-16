import constants as c
import numpy as np
import my_utils_Fashion 
import time
import torch


def main_Fashion_betas():

    device = my_utils_Fashion.set_seed_and_find_device()    
    train_loader, test_loader = my_utils_Fashion.get_Fashion_Dataloaders()
    
    accs_tot = []
    spks_tot = []

    st = time.time()
    for i, surr_name in enumerate(c.SURR_NAMES):

        print(f"\nSurrogate: {surr_name}\n")

        accuracies = []
        spks = []

        for beta in c.BETAS:
            net = my_utils_Fashion.Net_Betas(i, beta).to(device)
            print(f"\nBeta: {beta}")

            output = my_utils_Fashion.training(net=net,
                                                train_loader=train_loader,
                                                test_loader=test_loader, 
                                                device=device)
            del net    
            accuracies.append(np.array([output[:2]]))
            spks.append(np.array(output[5]))
        
        accs_tot.append(accuracies)
        spks_tot.append(spks)

    en = time.time()

    # Save data into .npy files
    np.save(f'Accuracy_vs_Sparsity/Accs_betas.npy', np.array(accs_tot)) 
    np.save(f'Accuracy_vs_Sparsity/Spks_tot_betas.npy', np.array(spks_tot))

    print(f'Run time: {(en-st)/60:.2f} min ({(en-st)/60/60:.2f}h)\n') 






def main_Fashion_surr_coeff():

    device = my_utils_Fashion.set_seed_and_find_device()    
    st = time.time()

    spks = []
    spks_tot = []
    acc_tot = []
    pot_tot = [] 
    # loop over the surrogate functions
    for index, surr_name in enumerate(c.SURR_NAMES):
        
        print(f'\nSurrogate: {c.SURR_NAMES[index]}')
        train_loader, test_loader = my_utils_Fashion.get_Fashion_Dataloaders()
        
        spks_loop = []
        spks_loop_tot = []
        accuracies = []
        potentials = []

        # loop over the slopes of each surrogate function
        for j in range(len(c.SURR_SLOPES[surr_name])):    
            
            coeff = c.SURR_SLOPES[surr_name][j]
            net = my_utils_Fashion.Net(index, coeff).to(device)

            output = my_utils_Fashion.training(net=net,
                                               train_loader=train_loader, 
                                               test_loader=test_loader,
                                               device=device)
            del net    
            
            # compute the mean over the last 50 epochs for the tot prob of spike
            spks_tot_new = np.mean(output[5][-50:][:], axis=0)

            spks_loop.append(np.array(output[5]))                
            spks_loop_tot.append(spks_tot_new)
            accuracies.append(np.array([output[:2]]))
            potentials.append(np.array([output[-1]]))

        spks.append(spks_loop)
        spks_tot.append(spks_loop_tot)
        acc_tot.append(accuracies)
        pot_tot.append(potentials)


    # Save data into .npy files
    np.save(f'Diff_Surrogates2/Spks_layers.npy', np.array(spks))
    np.save(f'Diff_Surrogates2/Spks_tot_overall.npy', spks_tot)
    np.save(f'Diff_Surrogates2/Accs.npy', np.array(acc_tot)) 
    np.save(f'Diff_Surrogates2/Potential.npy', np.array(pot_tot)) 


    en = time.time()
    print(f'Run time: {(en-st)/60:.2f} min ({(en-st)/60/60:.2f}h)\n') 



if __name__ == '__main__':
    main_Fashion_surr_coeff()