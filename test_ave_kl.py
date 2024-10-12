from synthetic_gmm import calc_average_kl_gap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def main():
    K = 2  
    N_k = 50  
    J = 2  
    L = 2  
    d = 2  # dimension
    n_samples = 200  # sample size of synthetic data

    plt.figure(figsize=(20, 6))

    it_values = range(2, 16)
    kl_gaps = []
    kl_gap_stds = []

    for K in tqdm(it_values):
        ROUND = 100
        kl_results = [calc_average_kl_gap(K, J, L, N_k, d, n_samples) for _ in range(ROUND)]
        kl_gap = np.mean(kl_results)
        kl_gap_std = np.std(kl_results)

        kl_gaps.append(kl_gap)
        kl_gap_stds.append(kl_gap_std)

    
    A1 = plt.subplot(1,3,1)
    A1.plot(it_values, kl_gaps, 'bD-', markeredgewidth=3,markeredgecolor='b', markersize=2, markerfacecolor='none', linewidth=4)
    A1.fill_between(it_values, np.array(kl_gaps) - np.array(kl_gap_stds), 
                np.array(kl_gaps) + np.array(kl_gap_stds), color='blue', alpha=0.2)
    A1.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.title('KL Gap vs K', fontsize=20)
    plt.xlabel('K', fontsize=18)
    plt.ylabel('KL Gap', fontsize=18)
    plt.xticks(it_values)  
    


    K = 2  
    N_k = 50  
    J = 2  
    L = 2  
    d = 2  # dimension
    n_samples = 200  # sample size of synthetic data
    kl_gaps = []
    kl_gap_stds = []

    for J in tqdm(it_values):
        ROUND = 100
        kl_results = [calc_average_kl_gap(K, J, L, N_k, d, n_samples) for _ in range(ROUND)]
        kl_gap = np.mean(kl_results)
        kl_gap_std = np.std(kl_results)

        kl_gaps.append(kl_gap)
        kl_gap_stds.append(kl_gap_std)

    
    A2 = plt.subplot(1,3,2)
    A2.plot(it_values, kl_gaps, 'bD-', markeredgewidth=3,markeredgecolor='b', markersize=2, markerfacecolor='none', linewidth=4)
    A2.fill_between(it_values, np.array(kl_gaps) - np.array(kl_gap_stds), 
                np.array(kl_gaps) + np.array(kl_gap_stds), color='blue', alpha=0.2)
    A2.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.title('KL Gap vs J', fontsize=20)
    plt.xlabel('J', fontsize=18)
    plt.ylabel('KL Gap', fontsize=18)
    plt.xticks(it_values)  



    #it_values = range(2, 11)
    K = 2  
    N_k = 50  
    J = 2  
    L = 2  
    d = 2  # dimension
    n_samples = 200  # sample size of synthetic data
    kl_gaps = []
    kl_gap_stds = []

    for L in tqdm(it_values):
        ROUND = 100
        kl_results = [calc_average_kl_gap(K, J, L, N_k, d, n_samples) for _ in range(ROUND)]
        kl_gap = np.mean(kl_results)
        kl_gap_std = np.std(kl_results)

        kl_gaps.append(kl_gap)
        kl_gap_stds.append(kl_gap_std)

    A3 = plt.subplot(1,3,3)
    A3.plot(it_values, kl_gaps, 'bD-', markeredgewidth=3,markeredgecolor='b', markersize=2, markerfacecolor='none', linewidth=4)
    A3.fill_between(it_values, np.array(kl_gaps) - np.array(kl_gap_stds), 
                np.array(kl_gaps) + np.array(kl_gap_stds), color='blue', alpha=0.2)
    A3.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.title('KL Gap vs L', fontsize=20)
    plt.xlabel('L', fontsize=18)
    plt.ylabel('KL Gap', fontsize=18)
    plt.xticks(it_values)  


    plt.subplots_adjust(wspace=0.4)

    plt.savefig('./kl_gap.pdf', format='pdf')  
    plt.show()  


if __name__ == "__main__":
    main()