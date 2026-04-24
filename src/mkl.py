import numpy as np
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, root_mean_squared_error, mean_squared_error, r2_score

def mkl_objective(weights, K_list, y):
    """
    Constructs a combined kernel and finds the best alpha for KernelRidge
    using a quick grid search. Returns the cross-validated RMSE.
    """
    # Build the combined kernel: sum(w_m * K_m)
    w_norm = weights / (np.sum(weights) + 1e-10)
    
    K_combined = np.zeros_like(K_list[0])
    for w, K in zip(w_norm, K_list):
        K_combined += w * K
        
    # Quick grid search for alpha
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
    # CV=5 internally to find best alpha for THESE weights
    grid_search = GridSearchCV(
        KernelRidge(kernel='precomputed'),
        param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(K_combined, y)
    
    # Return positive RMSE (minimize wants to go down)
    return -grid_search.best_score_

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

    print(f"Optimizing MKL weights on {len(train_idx)} training samples (with alpha tuning)...")
    
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
        
    # Tune alpha one last time for the final combined training kernel
    param_grid = {'alpha': np.logspace(-4, 2, 13)} # Finer grid for final model
    grid_search = GridSearchCV(
        KernelRidge(kernel='precomputed'),
        param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(K_combined_train, y_train)
    best_alpha = grid_search.best_params_['alpha']
    print(f"Best Final Alpha: {best_alpha:.4f}")

    final_regr = grid_search.best_estimator_
    y_pred_test = final_regr.predict(K_combined_test)
    
    test_metrics = {
        'MSE': mean_squared_error(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'R2': r2_score(y_test, y_pred_test),
        'Best_Alpha': best_alpha
    }
    
    # Sort by weight descending
    sorted_drivers = dict(sorted(pathway_importances.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_drivers, test_metrics, y_test, y_pred_test
