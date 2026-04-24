import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

def normalize_kernel(K):
    """
    Normalizes a kernel matrix so that all diagonal elements are 1.
    Formula: K_norm(i, j) = K(i, j) / sqrt(K(i, i) * K(j, j))
    """
    d = np.diag(K).copy()
    d[d == 0] = 1e-10 
    D_inv_sqrt = 1.0 / np.sqrt(d)
    K_norm = K * D_inv_sqrt[:, np.newaxis] * D_inv_sqrt[np.newaxis, :]
    return K_norm

def compute_kernels(df, feature_dict, kernel_type='linear', gamma=None):
    """
    Computes an N x N kernel matrix for each feature subset in feature_dict.
    """
    kernels = {}
    for name, genes in feature_dict.items():
        valid_genes = [g for g in genes if g in df.columns]
        if not valid_genes:
            continue
            
        X_subset = df[valid_genes].values
        
        if kernel_type == 'linear':
            K = linear_kernel(X_subset)
        elif kernel_type == 'rbf':
            K = rbf_kernel(X_subset, gamma=gamma)
            
        kernels[name] = K
        
    return kernels

def normalize_all_kernels(kernels_dict):
    """
    Iterates through a dictionary of kernels and normalizes each one.
    """
    return {name: normalize_kernel(K) for name, K in kernels_dict.items()}
