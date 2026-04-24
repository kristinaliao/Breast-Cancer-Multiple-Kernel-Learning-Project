import numpy as np
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, root_mean_squared_error, mean_squared_error, r2_score

def mkl_objective(weights, K_list, y):
    """
    Constructs a combined kernel using the given weights, trains a 
    Kernel Ridge Regression model, and returns the Cross-Validated RMSE.
    """
    # Build the combined kernel: sum(w_m * K_m)
    # Ensure weights are normalized for the objective
    w_norm = weights / (np.sum(weights) + 1e-10)
    
    K_combined = np.zeros_like(K_list[0])
    for w, K in zip(w_norm, K_list):
        K_combined += w * K
        
    # Initialize the base learner
    regr = KernelRidge(kernel='precomputed', alpha=1.0)
    
    # Evaluate using 5-Fold Cross Validation
    rmse_scorer = make_scorer(root_mean_squared_error, response_method='predict')
    scores = cross_val_score(regr, K_combined, y, cv=5, scoring=rmse_scorer)
    
    return np.mean(scores)

def run_meta_learner(normalized_kernels, y_target, train_idx, test_idx):
    """
    Optimizes kernel weights on training set and evaluates on test set.
    """
    kernel_names = list(normalized_kernels.keys())
    K_list = list(normalized_kernels.values())
    
    # Ensure y is a 1D numpy array
    y_target_arr = y_target.values.ravel() if hasattr(y_target, 'values') else np.array(y_target).ravel()
    
    y_train = y_target_arr[train_idx]
    y_test = y_target_arr[test_idx]
    
    # Slice kernels for training
    K_list_train = [K[np.ix_(train_idx, train_idx)] for K in K_list]
    
    n_kernels = len(K_list)
    initial_weights = np.ones(n_kernels) / n_kernels
    bounds = [(0.0, 1.0) for _ in range(n_kernels)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w**2) - 1.0})

    print(f"Optimizing MKL weights on {len(train_idx)} training samples...")
    
    result = minimize(
        mkl_objective, 
        initial_weights, 
        args=(K_list_train, y_train), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'disp': False, 'maxiter': 200}
    )

    optimal_weights = result.x
    normalized_weights = optimal_weights / np.sum(optimal_weights)
    pathway_importances = {name: weight for name, weight in zip(kernel_names, normalized_weights)}
    
    # Final Evaluation on Test Set
    print(f"Evaluating MKL on {len(test_idx)} hold-out samples...")
    
    # Kernel for testing: Rows = Test, Cols = Train
    K_list_test = [K[np.ix_(test_idx, train_idx)] for K in K_list]
    
    # Build final combined kernels
    K_combined_train = np.zeros_like(K_list_train[0])
    for w, K in zip(normalized_weights, K_list_train):
        K_combined_train += w * K
        
    K_combined_test = np.zeros_like(K_list_test[0])
    for w, K in zip(normalized_weights, K_list_test):
        K_combined_test += w * K
        
    final_regr = KernelRidge(kernel='precomputed', alpha=1.0)
    final_regr.fit(K_combined_train, y_train)
    y_pred_test = final_regr.predict(K_combined_test)
    
    test_metrics = {
        'MSE': mean_squared_error(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'R2': r2_score(y_test, y_pred_test)
    }
    
    # Sort by weight descending
    sorted_drivers = dict(sorted(pathway_importances.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_drivers, test_metrics
