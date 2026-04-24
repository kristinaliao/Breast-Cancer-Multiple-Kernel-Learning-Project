import pandas as pd
import numpy as np
from data_processing import load_and_align_data, preprocess_data, parse_gmt_and_map
from baselines import run_baselines
from kernels import compute_kernels, normalize_all_kernels, normalize_kernel
from mkl import run_meta_learner
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

def main():
    # File paths
    clinical_path = 'X_clinical.csv'
    rppa_path = 'merged_RPPA.csv'
    mrna_path = 'merged_mRNA_TPM.csv'
    target_path = 'target_vector.csv'
    gmt_path = 'hallmark.gmt'

    print("--- Loading and Aligning Data ---")
    X_clinical, X_rppa, df_mrna, y = load_and_align_data(
        clinical_path, rppa_path, mrna_path, target_path
    )
    print(f"Data aligned. Samples: {len(y)}")

    print("\n--- Preprocessing Data ---")
    df_mrna_filtered, X_rppa_imputed = preprocess_data(df_mrna, X_rppa)
    print(f"mRNA features after filtering: {df_mrna_filtered.shape[1]}")
    print(f"RPPA features after imputation: {X_rppa_imputed.shape[1]}")


    print("\n--- Running Baseline Models (RF & SVR) ---")
    try: 
        with open("baseline_results.txt", "x") as f:
            y_target = y['ProliferationScore']
            baseline_results = run_baselines(X_clinical, X_rppa_imputed, df_mrna_filtered, y_target)
            for model, metrics in baseline_results.items():
                f.write(f"\n{model} Results:")
                f.write(f"  MSE: {metrics['MSE']:.4f}")
                f.write(f"  R2:  {metrics['R2']:.4f}")
    except FileExistsError:
        with open("baseline_results.txt", "r") as f:
            baseline_results = f.read()
            print(baseline_results)

    print("\n--- Kernel Computation & Normalization ---")
    # 1. Parse GMT for hallmark pathways
    hallmark_dict = parse_gmt_and_map(gmt_path, list(df_mrna_filtered.columns))

    ## REMOVE cell-cycle/proliferation proxy kernels
    leakage_kernels = [
        'HALLMARK_SPERMATOGENESIS', 
        'HALLMARK_G2M_CHECKPOINT', 
        'HALLMARK_E2F_TARGETS', 
        'HALLMARK_MITOTIC_SPINDLE',
        'HALLMARK_DNA_REPAIR'
    ]

    clean_hallmark_dict = {k: v for k, v in hallmark_dict.items() if k not in leakage_kernels}
    
    # 2. Define custom biological lists
    adhesion_genes = ['CDH1', 'CTNNA1', 'CTNNB1', 'CTNND1']
    akt_genes = ['PTEN', 'PIK3CA', 'AKT1', 'AKT2', 'AKT3', 'INPP4B', 'EGFR', 'ERBB2', 'STAT3']
    tf_genes = ['FOXA1', 'GATA3', 'RUNX1', 'TBX3', 'ESR1']
    
    custom_lists = {
        'ILC_Adhesion': adhesion_genes,
        'ILC_AKT_Pathway': akt_genes,
        'ILC_TF_Drivers': tf_genes
    }
    
    # Combine pathway dictionaries
    pathways_dict = {**custom_lists, **clean_hallmark_dict}
    
    # 3. Compute mRNA pathway kernels
    mRNA_kernels = compute_kernels(df_mrna_filtered, pathways_dict, kernel_type='linear')
    normalized_kernels = normalize_all_kernels(mRNA_kernels)
    
    # 4. Add Global Clinical and RPPA Kernels
    K_clinical = linear_kernel(X_clinical.values)
    normalized_kernels['Clinical_Global'] = normalize_kernel(K_clinical)
    
    K_rppa = rbf_kernel(X_rppa_imputed.values)
    normalized_kernels['RPPA_Global'] = normalize_kernel(K_rppa)
    
    print(f"Total kernels prepared: {len(normalized_kernels)}")

    print("\n--- Running Meta-Learner (Multiple Kernel Learning) ---")
    sorted_drivers, final_rmse = run_meta_learner(normalized_kernels, y_target)

    print(f"\nOptimization Finished. Final CV RMSE: {final_rmse:.4f}")
    print("\n--- Top Biological Drivers of ILC Proliferation ---")
    for pathway, weight in sorted_drivers.items():
        if weight > 0.01:
            print(f"{pathway}: {weight:.4f}")

if __name__ == "__main__":
    main()
