import constants as c
import numpy as np
import my_utils_Fashion 
import time
def main_Fashion():
    device = my_utils_Fashion.set_seed_and_find_device()    
    train_loader, test_loader = my_utils_Fashion.get_Fashion_Dataloaders()
    betas = [0.2, 0.5, 0.8, 0.9, 0.93, 0.96, 0.99]
    
    accs_tot = []
    spks_tot = []

    st = time.time()
    for i, surr_name in enumerate(c.SURR_NAMES):

        print(f"\nSurrogate: {surr_name}\n")

        accuracies = []
        spks = []

        for beta in betas:
            net = my_utils_Fashion.Net2(surr_info=[i, beta]).to(device)
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


    print(f'Run time: {(en-st)/60:.2f} min\n') 



if __name__ == '__main__':
    main_Fashion()