import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def run_baselines(X_train, X_test, y_train, y_test):
    """
    Performs hyperparameter tuning on training data and evaluates on hold-out test set.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # 1. Random Forest with Grid Search
    print("Tuning Random Forest...")
    rf_param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42), 
        rf_param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    rf_grid.fit(X_train_scaled, y_train)
    best_rf = rf_grid.best_estimator_
    
    y_pred_rf = best_rf.predict(X_test_scaled)
    results['Random Forest'] = {
        'MSE': mean_squared_error(y_test, y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'R2': r2_score(y_test, y_pred_rf),
        'Best_Params': rf_grid.best_params_
    }

    # 2. SVR with Grid Search
    print("Tuning SVR...")
    svr_param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.5],
        'gamma': ['scale', 'auto']
    }
    svr_grid = GridSearchCV(
        SVR(kernel='rbf'), 
        svr_param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    svr_grid.fit(X_train_scaled, y_train)
    best_svr = svr_grid.best_estimator_
    
    y_pred_svr = best_svr.predict(X_test_scaled)
    results['SVR'] = {
        'MSE': mean_squared_error(y_test, y_pred_svr),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svr)),
        'R2': r2_score(y_test, y_pred_svr),
        'Best_Params': svr_grid.best_params_
    }

    return results
