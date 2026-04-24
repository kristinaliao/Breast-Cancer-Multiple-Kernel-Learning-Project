import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def run_baselines(X_clinical, X_rppa, df_mrna, y_target):
    """
    Combines data modalities and runs Random Forest and SVR baselines.
    """
    # Combine all modalities
    X_combined = pd.concat([X_clinical, X_rppa, df_mrna], axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_target, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    results['Random Forest'] = {
        'MSE': mean_squared_error(y_test, y_pred_rf),
        'R2': r2_score(y_test, y_pred_rf)
    }

    # SVR
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    y_pred_svr = svr.predict(X_test_scaled)
    results['SVR'] = {
        'MSE': mean_squared_error(y_test, y_pred_svr),
        'R2': r2_score(y_test, y_pred_svr)
    }

    return results
