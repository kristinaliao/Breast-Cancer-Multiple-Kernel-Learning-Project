import pandas as pd
import numpy as np
import argparse
import sys
from src.data_processing import load_and_align_data, preprocess_data, parse_gmt_and_map
from src.baselines import run_baselines
from src.kernels import compute_kernels, normalize_all_kernels, normalize_kernel
from src.mkl import run_meta_learner
from src.biological_lists import CUSTOM_LISTS, LEAKAGE_KERNELS, RPPA_PATHWAYS
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.model_selection import train_test_split

def run_mkl_pipeline(normalized_kernels, y_target, train_idx, test_idx, silent=False):
    """Helper to run MKL and return results."""
    if not silent:
        print(f"Running MKL with {len(normalized_kernels)} kernels...")
    sorted_drivers, metrics, y_test, y_pred = run_meta_learner(
        normalized_kernels, y_target, train_idx, test_idx
    )
    if not silent:
        print("Top 10 Biological Drivers:")
        for i, (pathway, weight) in enumerate(sorted_drivers.items()):
            if i >= 10: break
            print(f"  {i+1}. {pathway}: {weight:.4f}")
    return sorted_drivers, metrics

def main():
    parser = argparse.ArgumentParser(description='Run ILC Proliferation Prediction Pipeline')
    parser.add_argument('--mode', type=str, choices=['baseline', 'mkl', 'both'], default='both',
                        help='Which models to run in standard mode (default: both)')
    parser.add_argument('--experiment', type=str, choices=['pruning', 'ablation', 'bootstrapping'],
                        help='Run a specific experiment')
    args = parser.parse_args()

    # File paths
    clinical_path = 'data/X_clinical.csv'
    rppa_path = 'data/merged_RPPA.csv'
    mrna_path = 'data/merged_mRNA_TPM.csv'
    target_path = 'data/target_vector.csv'
    gmt_path = 'data/hallmark.gmt'

    print("--- Loading and Aligning Data ---")
    X_clinical, X_rppa, df_mrna, y = load_and_align_data(
        clinical_path, rppa_path, mrna_path, target_path
    )
    y_target = y['ProliferationScore']
    
    print("\n--- Preprocessing Data ---")
    df_mrna_filtered, X_rppa_imputed = preprocess_data(df_mrna, X_rppa)
    
    # 1. Create a consistent Train/Test split using indices to ensure both 
    # Baselines and MKL are evaluated on the exact same hold-out patients.
    indices = np.arange(len(y_target))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # If an experiment is specified, run it and exit.
    if args.experiment:
        run_experiment(args.experiment, X_clinical, X_rppa_imputed, df_mrna_filtered, y_target, train_idx, test_idx, gmt_path)
        return
    
    results_file = "evaluation_results.txt"
    # Initialize the results file for this run
    with open(results_file, "w") as f:
        f.write(f"--- ILC Proliferation Score Prediction Results (Mode: {args.mode}) ---\n")

    # 2. Run Baselines (Random Forest & SVR)
    if args.mode in ['baseline', 'both']:
        # Prepare tabular data by concatenating all modalities
        X_combined = pd.concat([X_clinical, X_rppa_imputed, df_mrna_filtered], axis=1)
        X_train_baseline = X_combined.iloc[train_idx]
        X_test_baseline = X_combined.iloc[test_idx]
        y_train_baseline = y_target.iloc[train_idx]
        y_test_baseline = y_target.iloc[test_idx]

        print("\n--- Running Baseline Models (Tuning on Train, Evaluating on Test) ---")
        baseline_results = run_baselines(X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline)
        
        with open(results_file, "a") as f:
            f.write("\n--- Baseline Hold-out Test Results ---\n")
            for model, metrics in baseline_results.items():
                f.write(f"\n{model} Results:\n")
                f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
                f.write(f"  R2:   {metrics['R2']:.4f}\n")
                f.write(f"  Best Params: {metrics['Best_Params']}\n")

    # 3. Run Multiple Kernel Learning (MKL)
    if args.mode in ['mkl', 'both']:
        print("\n--- Kernel Computation & Normalization ---")
        # Prepares pathway-based kernels (mRNA linear, RPPA RBF, Clinical linear)
        kernels = prepare_all_kernels(X_clinical, X_rppa_imputed, df_mrna_filtered, gmt_path)
        
        print(f"Total kernels prepared: {len(kernels)}")

        print("\n--- Running Meta-Learner (Optimizing Weights on Train, Evaluating on Test) ---")
        # Optimizes kernel weights via inner CV on train set, then evaluates once on test set
        sorted_drivers, mkl_metrics = run_mkl_pipeline(kernels, y_target, train_idx, test_idx)
        
        with open(results_file, "a") as f:
            f.write(f"\n--- MKL Hold-out Test Results ---\nRMSE: {mkl_metrics['RMSE']:.4f}, R2: {mkl_metrics['R2']:.4f}\n")
            f.write("\nTop 10 Biological Drivers:\n")
            for i, (pathway, weight) in enumerate(sorted_drivers.items()):
                if i >= 10: break
                f.write(f"  {i+1}. {pathway}: {weight:.4f}\n")
        
        print(f"\nMKL Finished. Test RMSE: {mkl_metrics['RMSE']:.4f}")

def prepare_all_kernels(X_clinical, X_rppa, df_mrna, gmt_path):
    """
    Constructs and normalizes all pathway-based kernels for the MKL pipeline.
    
    This includes:
    1. mRNA pathway kernels: Linear kernels for Hallmark pathways and custom ILC lists.
    2. RPPA functional kernels: RBF kernels for targeted protein modules.
    3. Clinical kernel: A global linear kernel for clinical features.
    """
    hallmark_dict = parse_gmt_and_map(gmt_path, list(df_mrna.columns))
    clean_hallmark_dict = {k: v for k, v in hallmark_dict.items() if k not in LEAKAGE_KERNELS}
    pathways_dict = {**CUSTOM_LISTS, **clean_hallmark_dict}
    
    kernels = normalize_all_kernels(compute_kernels(df_mrna, pathways_dict, kernel_type='linear'))
    rppa_kernels = normalize_all_kernels(compute_kernels(X_rppa, RPPA_PATHWAYS, kernel_type='rbf'))
    kernels.update(rppa_kernels)
    kernels['Clinical_Global'] = normalize_kernel(linear_kernel(X_clinical.values))
    return kernels

def run_experiment(exp_type, X_clinical, X_rppa, df_mrna, y_target, train_idx, test_idx, gmt_path):
    """
    Executes a specific research experiment to validate model stability and feature importance.
    
    Available Experiments:
    - pruning: Evaluates performance using only a hand-picked set of 15 relevant kernels.
    - ablation: Compares mRNA-only vs. RPPA-only vs. Combined model performance.
    - bootstrapping: Runs 30 iterations of sampling with replacement to check weight consistency.
    """
    print(f"\n=== Running Experiment: {exp_type.upper()} ===")
    all_kernels = prepare_all_kernels(X_clinical, X_rppa, df_mrna, gmt_path)
    
    pruned_list = [
        'Clinical_Global', 'ILC_AKT_Pathway', 'ILC_TF_Drivers', 'ILC_Adhesion',
        'RPPA_PI3K_AKT_mTOR', 'RPPA_Receptors_RTK', 'RPPA_MAPK_ERK', 'RPPA_Adhesion_EMT',
        'HALLMARK_KRAS_SIGNALING_DN', 'HALLMARK_ESTROGEN_RESPONSE_EARLY', 
        'HALLMARK_ESTROGEN_RESPONSE_LATE', 'HALLMARK_APICAL_JUNCTION', 
        'HALLMARK_GLYCOLYSIS', 'HALLMARK_HYPOXIA', 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION'
    ]

    with open("experiment_results.txt", "a") as f:
        f.write(f"\n\n{'='*20} EXPERIMENT: {exp_type.upper()} {'='*20}\n")

    if exp_type == 'pruning':
        pruned_kernels = {k: v for k, v in all_kernels.items() if k in pruned_list}
        
        print("Running Full MKL (Benchmark)...")
        _, full_metrics = run_mkl_pipeline(all_kernels, y_target, train_idx, test_idx)
        print("Running Pruned MKL...")
        _, pruned_metrics = run_mkl_pipeline(pruned_kernels, y_target, train_idx, test_idx)
        
        with open("experiment_results.txt", "a") as f:
            f.write(f"Full Model RMSE:   {full_metrics['RMSE']:.4f}\n")
            f.write(f"Pruned Model RMSE: {pruned_metrics['RMSE']:.4f}\n")

    elif exp_type == 'ablation':
        mrna_kernels = {k: v for k, v in all_kernels.items() if k.startswith('HALLMARK_') or k.startswith('ILC_')}
        rppa_kernels = {k: v for k, v in all_kernels.items() if k.startswith('RPPA_')}
        
        print("Running mRNA-only MKL...")
        _, mrna_metrics = run_mkl_pipeline(mrna_kernels, y_target, train_idx, test_idx)
        print("Running RPPA-only MKL...")
        _, rppa_metrics = run_mkl_pipeline(rppa_kernels, y_target, train_idx, test_idx)
        print("Running Combined MKL...")
        _, both_metrics = run_mkl_pipeline({**mrna_kernels, **rppa_kernels}, y_target, train_idx, test_idx)

        with open("experiment_results.txt", "a") as f:
            f.write(f"mRNA-only RMSE: {mrna_metrics['RMSE']:.4f}\n")
            f.write(f"RPPA-only RMSE: {rppa_metrics['RMSE']:.4f}\n")
            f.write(f"Combined RMSE:  {both_metrics['RMSE']:.4f}\n")

    elif exp_type == 'bootstrapping':
        # Use ONLY pruned kernels for bootstrapping
        bootstrap_kernels = {k: v for k, v in all_kernels.items() if k in pruned_list}
        n_iterations = 30
        all_weights = []
        all_rmse = []
        kernel_names = list(bootstrap_kernels.keys())
        
        boot_file = "bootstrapping_results.txt"
        with open(boot_file, "w") as f:
            f.write(f"=== ILC Bootstrapping Experiment ({n_iterations} Iterations) ===\n")
            f.write(f"Kernels used: {', '.join(kernel_names)}\n\n")

        print(f"Starting Bootstrap ({n_iterations} iterations) on {len(kernel_names)} pruned kernels...")
        for i in range(n_iterations):
            # Sampling WITH REPLACEMENT from the training indices
            boot_train_idx = np.random.choice(train_idx, size=len(train_idx), replace=True)
            
            # Use original test_idx for consistent evaluation
            drivers, metrics = run_mkl_pipeline(bootstrap_kernels, y_target, boot_train_idx, test_idx, silent=True)
            
            # Display and Log top 10 for this specific bootstrap run
            print(f"  Iteration {i+1} Top Drivers:")
            with open(boot_file, "a") as f:
                f.write(f"Iteration {i+1} | RMSE: {metrics['RMSE']:.4f}\n")
                for j, (pathway, weight) in enumerate(drivers.items()):
                    if j >= 10: break
                    output_line = f"    {j+1}. {pathway}: {weight:.4f}"
                    print(output_line)
                    f.write(output_line + "\n")
                f.write("\n")
            
            # Map weights to consistent order
            weights = [drivers.get(name, 0) for name in kernel_names]
            all_weights.append(weights)
            all_rmse.append(metrics['RMSE'])
            if (i+1) % 10 == 0:
                print(f" Completed {i+1}/{n_iterations} iterations.")

        avg_weights = np.mean(all_weights, axis=0)
        std_weights = np.std(all_weights, axis=0)
        
        # Create a clean DataFrame for the final stability report
        df_stability = pd.DataFrame({
            'Pathway': kernel_names,
            'Mean_Weight': avg_weights,
            'Std_Deviation': std_weights
        }).sort_values(by='Mean_Weight', ascending=False)
        
        stability_report = df_stability.to_string(index=False)
        
        with open(boot_file, "a") as f:
            f.write("\n--- Final Bootstrap Stability Results ---\n")
            f.write(stability_report)
            f.write(f"\n\nBootstrap Avg RMSE: {np.mean(all_rmse):.4f} (+/- {np.std(all_rmse):.4f})\n")
            
        with open("experiment_results.txt", "a") as f:
            f.write(f"Bootstrap Avg RMSE: {np.mean(all_rmse):.4f} (+/- {np.std(all_rmse):.4f})\n")
            f.write("Full stability analysis saved to 'bootstrapping_results.txt'\n")
            
        print("\n--- Final Bootstrap Stability Results ---")
        print(stability_report)
        print(f"\nBootstrapping complete. Detailed results in 'bootstrapping_results.txt'.")

if __name__ == "__main__":
    main()
