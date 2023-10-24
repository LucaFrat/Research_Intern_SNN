import constants as c
import numpy as np
import my_utils_Fashion 
import time



def main_Fashion():

    device = my_utils_Fashion.set_seed_and_find_device()    
    train_loader, test_loader = my_utils_Fashion.get_Fashion_Dataloaders()

    # betas from 0.2 to 0.9 (0.1 step) + from 0.9 to 0.98 (0.02 step)
    betas = np.append(np.arange(c.BETA, 0.9, 0.1, dtype=float), np.arange(0.9, 1.0, 0.02))

    print(f"\nSurrogate: {c.SURR_NAMES[0]}\n")

    accuracies = []
    spks = []

    st = time.time()
    for beta in betas:
        net = my_utils_Fashion.Net(surr_func=c.SURR_FUNCTIONS[0], beta=beta, dataset=1).to(device)
        print(f"Beta: {beta}")

        output = my_utils_Fashion.training(net=net,
                                            train_loader=train_loader, 
                                            test_loader=test_loader, 
                                            device=device)
        del net    
        accuracies.append(np.array([output[:2]]))
        spks.append(np.array(output[5]))

    en = time.time()

    # Save data into .npy files
    np.save(f'Accuracy_vs_Sparsity/Accs_{c.SURR_NAMES[0]}_betas.npy', np.array(accuracies)) 
    np.save(f'Accuracy_vs_Sparsity/Spks_tot_{c.SURR_NAMES[0]}_betas.npy', np.array(spks))


    print(f'Run time: {(en-st)/60:.2f} min\n') 



if __name__ == '__main__':
    main_Fashion()