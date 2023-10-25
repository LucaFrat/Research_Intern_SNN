import constants as c
import numpy as np
import my_utils_Fashion 
import time



def main_Fashion():

    device = my_utils_Fashion.set_seed_and_find_device()    
    st = time.time()


    spks_tot = []
    # loop over the surrogate functions
    for index, surr_name in enumerate(c.SURR_NAMES):
        
        print(f'\nSurrogate: {c.SURR_NAMES[index]}\n')
        train_loader, test_loader = my_utils_Fashion.get_Fashion_Dataloaders()
        
        accuracies = []
        spks_1 = []
        spks_2 = []
        spks_3 = []

        # loop over the slopes of each surrogate function
        for j in range(len(c.SURR_SLOPES[surr_name])):    
            
            coeff = c.SURR_SLOPES[surr_name][j]
            net = my_utils_Fashion.Net([index, coeff]).to(device)

            output = my_utils_Fashion.training(net=net,
                                               train_loader=train_loader, 
                                               test_loader=test_loader,
                                               device=device)
            del net    
            
            # compute the mean over the last 75 epochs for the tot prob of spike
            spks_tot_new = np.mean(np.array(output[5])[-75:-1])
            spks_tot.append(spks_tot_new)

            accuracies.append(np.array([output[:2]]))
            spks_1.append(np.array(output[2]))
            spks_2.append(np.array(output[3]))
            spks_3.append(np.array(output[4]))                


        # Save data into .npy files
        np.save(f'Diff_Surrogates/Accs_{surr_name}.npy', np.array(accuracies)) 
        np.save(f'Diff_Surrogates/Spks_{surr_name}_layer1.npy', np.array(spks_1))
        np.save(f'Diff_Surrogates/Spks_{surr_name}_layer2.npy', np.array(spks_2))
        np.save(f'Diff_Surrogates/Spks_{surr_name}_layer3.npy', np.array(spks_3))

    np.save(f'Diff_Surrogates/Spks_tot_overall.npy', spks_tot)


    en = time.time()
    print(f'Run time: {(en-st)/60:.2f} min ({(en-st)/60/60:.2f}h)\n') 



if __name__ == '__main__':
    main_Fashion()