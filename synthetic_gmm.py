import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import entropy


def generate_ground_truth_gmm(K, J, d, gap=0):
    n_components = K + J  

    means = np.random.rand(n_components, d) * 10 + np.arange(n_components).reshape(-1, 1) * gap
    
    covariances = [1 * (np.random.rand(d, d) @ np.random.rand(d, d).T) for _ in range(n_components)]
    weights = np.ones(n_components) / n_components


    gmm_f = GaussianMixture(n_components=n_components)
    gmm_f.means_ = means
    gmm_f.covariances_ = covariances
    gmm_f.weights_ = weights
    
    return gmm_f


def sample_anchor_data(gmm, K, num_samples_per_component):
    means = gmm.means_[:K]
    covariances = gmm.covariances_[:K]
    anchor_data = []
    labels = []

    for k in range(K):
        samples = np.random.multivariate_normal(means[k], covariances[k], num_samples_per_component)
        anchor_data.append(samples)
        labels.append(np.full((num_samples_per_component,), k))  # add label
    anchor_data = np.vstack(anchor_data)
    labels = np.concatenate(labels)
    return np.column_stack((anchor_data, labels))  

def create_gmm_M(gmm_f, anchor_data, K, J, L, d):
    n_components = K + J + L
    gmm_M = GaussianMixture(n_components=n_components)

    means = np.zeros((n_components, d))
    covariances = np.zeros((n_components, d, d))

    # K part
    for k in range(K):
        component_data = anchor_data[anchor_data[:, -1] == k][:, :-1]  
        if component_data.size > 0:  
            means[k] = np.mean(component_data, axis=0)
            covariances[k] = np.cov(component_data, rowvar=False) + 1e-6 * np.eye(d)
        else:
            means[k] = np.random.rand(d) * 10
            covariances[k] = np.eye(d) + 1e-6 * np.eye(d)

    # J part
    means[K:K + J] = gmm_f.means_[K:K + J]
    covariances[K:K + J] = gmm_f.covariances_[K:K + J]

    # L part
    means[K + J:] = np.random.rand(L, d) * 10
    covariances[K + J:] = np.array([np.random.rand(d, d) @ np.random.rand(d, d).T for _ in range(L)])

    weights = np.ones(n_components) / n_components
    gmm_M.means_ = means
    gmm_M.covariances_ = covariances
    gmm_M.weights_ = weights
    
    return gmm_M

def sample_synthetic_data(gmm_M, n_samples):
    synthetic_data, _ = gmm_M.sample(n_samples)
    return synthetic_data

def fit_gmm(data, n_components, d):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data)
    return gmm

def sample_from_gmm(gmm, n_samples):
    samples, _ = gmm.sample(n_samples)
    return samples

def compute_kl_divergence(p_samples, q_samples):
    p_density, _ = np.histogramdd(p_samples, bins=50, density=True)
    q_density, _ = np.histogramdd(q_samples, bins=50, density=True)
    
    p_density = np.clip(p_density, 1e-10, None)
    q_density = np.clip(q_density, 1e-10, None)
    
    kl_divergence = entropy(p_density.ravel(), q_density.ravel())
    return kl_divergence

def one_round_KL_test(anchor_gmm, synthetic_gmm, ground_truth_gmm):
    # re-sampling
    ground_truth_samples = sample_from_gmm(ground_truth_gmm, n_samples=1000)
    anchor_samples = sample_from_gmm(anchor_gmm, n_samples=1000)
    synthetic_samples = sample_from_gmm(synthetic_gmm, n_samples=1000)

    # compute kl div
    kl_anchor_vs_gt = compute_kl_divergence(anchor_samples, ground_truth_samples)
    kl_synthetic_vs_gt = compute_kl_divergence(synthetic_samples, ground_truth_samples)

    return kl_anchor_vs_gt - kl_synthetic_vs_gt

def calc_average_kl_gap(K,J,L,N_k,d,n_samples): 


    # ground-truth params
    ground_truth_gmm = generate_ground_truth_gmm(K, J, d)

    # generate anchor data
    anchor_data = sample_anchor_data(ground_truth_gmm, K, N_k)

    # create gmm_M
    gmm_M = create_gmm_M(ground_truth_gmm, anchor_data, K, J, L, d)

    # sample synthetic data from gmm_M 
    synthetic_data = sample_synthetic_data(gmm_M, n_samples)

    anchor_gmm = fit_gmm(anchor_data[:, :-1], K, d)  # ignore the label col
    synthetic_gmm = fit_gmm(synthetic_data, K + J + L, d)

    ROUND = 100
    kl_gap = 0
    for i in range(ROUND):
        kl_gap += one_round_KL_test(anchor_gmm, synthetic_gmm, ground_truth_gmm)
    kl_gap /= ROUND

    return kl_gap

if __name__ == "__main__": 
    K = 2  
    N_k = 50  
    J = 6  
    L = 6  
    d = 2  # dimension
    n_samples = 200  # sample size of synthetic data
    calc_average_kl_gap(K, J, L, N_k, d, n_samples)
