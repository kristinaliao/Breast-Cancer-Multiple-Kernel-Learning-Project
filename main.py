import pandas as pd
import numpy as np
import argparse
from src.data_processing import load_and_align_data, preprocess_data, parse_gmt_and_map
from src.baselines import run_baselines
from src.kernels import compute_kernels, normalize_all_kernels, normalize_kernel
from src.mkl import run_meta_learner
from src.biological_lists import CUSTOM_LISTS, LEAKAGE_KERNELS, RPPA_PATHWAYS
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description='Run ILC Proliferation Prediction Pipeline')
    parser.add_argument('--mode', type=str, choices=['baseline', 'mkl', 'both'], default='both',
                        help='Which models to run: baseline, mkl, or both (default: both)')
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
    print(f"Data aligned. Samples: {len(y)}")

    print("\n--- Preprocessing Data ---")
    df_mrna_filtered, X_rppa_imputed = preprocess_data(df_mrna, X_rppa)
    print(f"mRNA features after filtering: {df_mrna_filtered.shape[1]}")
    print(f"RPPA features after imputation: {X_rppa_imputed.shape[1]}")

    y_target = y['ProliferationScore']
    
    # 1. Create a consistent Train/Test split using indices
    indices = np.arange(len(y_target))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    results_file = "evaluation_results.txt"
    # Clear or initialize the results file
    with open(results_file, "w") as f:
        f.write(f"--- ILC Proliferation Score Prediction Results (Mode: {args.mode}) ---\n")

    # 2. Run Baselines
    if args.mode in ['baseline', 'both']:
        # Prepare data for baselines
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

    # 3. Run MKL
    if args.mode in ['mkl', 'both']:
        print("\n--- Kernel Computation & Normalization ---")
        # 1. Parse GMT for hallmark pathways
        hallmark_dict = parse_gmt_and_map(gmt_path, list(df_mrna_filtered.columns))

        ## REMOVE cell-cycle/proliferation proxy kernels
        clean_hallmark_dict = {k: v for k, v in hallmark_dict.items() if k not in LEAKAGE_KERNELS}
        
        # 2. Combine pathway dictionaries
        pathways_dict = {**CUSTOM_LISTS, **clean_hallmark_dict}
        
        # 3. Compute mRNA pathway kernels
        mRNA_kernels = compute_kernels(df_mrna_filtered, pathways_dict, kernel_type='linear')
        normalized_kernels = normalize_all_kernels(mRNA_kernels)
        
        # 4. Compute RPPA pathway kernels
        RPPA_kernels = compute_kernels(X_rppa_imputed, RPPA_PATHWAYS, kernel_type='rbf')
        normalized_RPPA_kernels = normalize_all_kernels(RPPA_kernels)
        normalized_kernels.update(normalized_RPPA_kernels)
        
        # 5. Add Global Clinical Kernel
        K_clinical = linear_kernel(X_clinical.values)
        normalized_kernels['Clinical_Global'] = normalize_kernel(K_clinical)
        
        print(f"Total kernels prepared: {len(normalized_kernels)}")

        print("\n--- Running Meta-Learner (Optimizing Weights on Train, Evaluating on Test) ---")
        sorted_drivers, mkl_test_metrics = run_meta_learner(normalized_kernels, y_target, train_idx, test_idx)

        with open(results_file, "a") as f:
            f.write("\n\n--- MKL Hold-out Test Results ---\n")
            f.write(f"  RMSE: {mkl_test_metrics['RMSE']:.4f}\n")
            f.write(f"  R2:   {mkl_test_metrics['R2']:.4f}\n")
            f.write("\nTop Biological Drivers (Weights):\n")
            for pathway, weight in sorted_drivers.items():
                if weight > 0.01:
                    f.write(f"  {pathway}: {weight:.4f}\n")

        print(f"\nOptimization Finished. Test RMSE: {mkl_test_metrics['RMSE']:.4f}")
        print("\n--- Top Biological Drivers of ILC Proliferation ---")
        for pathway, weight in sorted_drivers.items():
            if weight > 0.01:
                print(f"{pathway}: {weight:.4f}")

if __name__ == "__main__":
    main()
