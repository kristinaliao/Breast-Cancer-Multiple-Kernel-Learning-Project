import numpy as np
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, root_mean_squared_error

def mkl_objective(weights, K_list, y):
    """
    Constructs a combined kernel using the given weights, trains a 
    Kernel Ridge Regression model, and returns the Cross-Validated RMSE.
    """
    # Build the combined kernel: sum(w_m * K_m)
    K_combined = np.zeros_like(K_list[0])
    for w, K in zip(weights, K_list):
        K_combined += w * K
        
    # Initialize the base learner
    regr = KernelRidge(kernel='precomputed', alpha=1.0)
    
    # Evaluate using 5-Fold Cross Validation
    rmse_scorer = make_scorer(root_mean_squared_error)
    scores = cross_val_score(regr, K_combined, y, cv=5, scoring=rmse_scorer)
    
    # The optimizer tries to MINIMIZE this return value
    return np.mean(scores)

def run_meta_learner(normalized_kernels, y_target):
    """
    Optimizes kernel weights using SLSQP to minimize CV RMSE.
    """
    kernel_names = list(normalized_kernels.keys())
    K_list = list(normalized_kernels.values())
    
    # Ensure y is a 1D numpy array
    y_target_arr = y_target.values.ravel() if hasattr(y_target, 'values') else np.array(y_target).ravel()
    
    n_kernels = len(K_list)
    initial_weights = np.ones(n_kernels) / n_kernels
    bounds = [(0.0, 1.0) for _ in range(n_kernels)]
    # constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w**2) - 1.0})

    print(f"Starting Meta-Learner Optimization with {n_kernels} kernels...")
    
    result = minimize(
        mkl_objective, 
        initial_weights, 
        args=(K_list, y_target_arr), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'disp': True, 'maxiter': 200}
    )

    optimal_weights = result.x
    normalized_optimal_weights = optimal_weights / np.sum(optimal_weights)
    pathway_importances = {name: weight for name, weight in zip(kernel_names, normalized_optimal_weights)}
    
    # Sort by weight descending
    sorted_drivers = dict(sorted(pathway_importances.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_drivers, result.fun
