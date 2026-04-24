import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def load_and_align_data(clinical_path, rppa_path, mrna_path, target_path):
    """
    Loads all data modalities and aligns them by Case_ID.
    """
    y = pd.read_csv(target_path)
    y = y[['Case.ID', 'ProliferationScore']].rename(columns={'Case.ID': 'Case_ID'})
    
    X_clinical = pd.read_csv(clinical_path)
    X_rppa = pd.read_csv(rppa_path)
    df_mrna = pd.read_csv(mrna_path)

    # Rename first columns to Case_ID for consistency
    X_rppa = X_rppa.rename(columns={X_rppa.columns[0]: 'Case_ID'})
    df_mrna = df_mrna.rename(columns={df_mrna.columns[0]: 'Case_ID'})

    # Set indices
    y.set_index('Case_ID', inplace=True)
    X_clinical.set_index('Case_ID', inplace=True)
    X_rppa.set_index('Case_ID', inplace=True)
    df_mrna.set_index('Case_ID', inplace=True)

    # Align by master IDs from target vector
    master_IDs = y.index.tolist()
    X_clinical = X_clinical.loc[master_IDs]
    X_rppa = X_rppa.loc[master_IDs]
    df_mrna = df_mrna.loc[master_IDs]

    return X_clinical, X_rppa, df_mrna, y

def preprocess_data(df_mrna, X_rppa, std_threshold=0.1):
    """
    Performs log transformation, variance filtering, and imputation.
    """
    # mRNA Preprocessing
    df_mrna_log = np.log2(df_mrna + 1)
    # Filter by standard deviation
    final_mrna = df_mrna_log.loc[:, df_mrna_log.std() > std_threshold]
    
    # Proliferation leakage genes to drop
    leakage_genes = [
        'BIRC5', 'CCNB1', 'CDC20', 'CEP55', 'MKI67', 'NDC80', 
        'NUF2', 'PTTG1', 'RRM2', 'TYMS', 'UBE2C', 
        'CENPF', 'EXO1', 'KIF2C', 'MELK', 'MYBL2', 'ORC6L', 'UBE2T'
    ]
    df_mrna_filtered = final_mrna.drop(columns=leakage_genes, errors='ignore')

    # RPPA Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_rppa_imputed = pd.DataFrame(
        imputer.fit_transform(X_rppa), 
        columns=X_rppa.columns, 
        index=X_rppa.index
    )

    return df_mrna_filtered, X_rppa_imputed

def parse_gmt_and_map(gmt_filepath, available_genes):
    """
    Parses a .gmt file and maps the genes to the available columns in df_mrna.
    """
    pathway_dict = {}
    available_set = set(available_genes)
    
    with open(gmt_filepath, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            pathway_name = parts[0]
            # parts[1] is the URL/description, parts[2:] are the genes
            genes = set(parts[2:])
            
            # Intersection prevents KeyErrors later when slicing df_mrna
            valid_genes = list(genes.intersection(available_set))
            
            if valid_genes:
                pathway_dict[pathway_name] = valid_genes
                
    return pathway_dict
