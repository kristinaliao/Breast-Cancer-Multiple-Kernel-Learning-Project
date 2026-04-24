import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# 1. Define your RPPA functional modules based on exact column names
rppa_pathways = {
    # 1. PI3K/AKT/mTOR Survival & Metabolism
    # Captures the core metabolic engine and high AKT activation state.
    'RPPA_PI3K_AKT_mTOR': [
        'AKT', 'AKT_pS473', 'AKT_pT308', 'Akt2', 'Akt2_pS474', 'PTEN', 'INPP4B', 
        'PI3KP110ALPHA', 'PI3KP85', 'PI3K-p110-b', 'MTOR', 'MTOR_pS2448', 
        'TUBERIN', 'TUBERIN_pT1462', 'TSC1', 'P70S6K1', 'P70S6K_pT389', 
        'S6', 'S6_pS235S236', 'S6_pS240S244', '4EBP1', '4EBP1_pS65', 
        '4EBP1_pT37T46', '4EBP1_pT70', 'PRAS40', 'PRAS40_pT246'
    ],

    # 2. Hormone Receptors & Receptor Tyrosine Kinases (RTKs)
    # The upstream "fuel" sensors, heavily weighted toward ER/PR and HER family.
    'RPPA_Receptors_RTK': [
        'ERALPHA', 'ERALPHA_pS118', 'PR', 'AR', 'EGFR', 'EGFR_pY1068', 
        'EGFR_pY1173', 'HER2', 'HER2_pY1248', 'HER3', 'HER3_pY1289', 
        'IGFRb', 'IGF1R_pY1135Y1136', 'CMET', 'CMET_pY1235'
    ],

    # 3. Adhesion, EMT & Structure
    # Crucial for isolating the structural breakdown and E-cadherin loss.
    'RPPA_Adhesion_EMT': [
        'ECADHERIN', 'NCADHERIN', 'PCADHERIN', 'BETACATENIN', 
        'b-Catenin_pT41_S45', 'FIBRONECTIN', 'FN14', 'SNAIL', 'ZEB1', 
        'CLAUDIN7', 'COLLAGENVI', 'CAVEOLIN1'
    ],

    # 4. MAPK/ERK Proliferation Cascade
    # The classic growth cascade (often down-regulated or bypassed in lobular subtypes).
    'RPPA_MAPK_ERK': [
        'ARAF', 'ARAF_pS299', 'BRAF', 'BRAF_pS445', 'CRAF', 'CRAF_pS338', 
        'MEK1', 'MEK1_pS217S221', 'MEK2', 'MAPK_pT202Y204', 'p44-42-MAPK', 
        'P38MAPK', 'P38_pT180Y182', 'p38-a', 'JNK2', 'JNK_pT183Y185'
    ],

    # 5. Cell Cycle & DNA Repair
    # The mechanical "exhaust" and checkpoint monitors of cell division.
    'RPPA_CellCycle_Repair': [
        'CDK1', 'CDK1_pT14', 'CDK1_pY15', 'CYCLINB1', 'CYCLIND1', 'CYCLINE1', 
        'CYCLINE2', 'P16INK4A', 'P21', 'P27', 'P27_pT157', 'P27_pT198', 
        'RB', 'RB_pS807S811', 'P53', 'ATM', 'ATM_pS1981', 'ATR', 'ATR_pS428', 
        'CHK1', 'CHK1_pS296', 'CHK1_pS345', 'CHK2', 'CHK2_pT68', 
        'H2AX_pS139', 'H2AX_pS140'
    ],

    # 6. Apoptosis & Survival
    # Measures the tumor's resistance to cell death.
    'RPPA_Apoptosis': [
        'BAX', 'BAK', 'BAD_pS112', 'BCL2', 'BCL2A1', 'BCLXL', 'BIM', 'BID', 
        'Mcl-1', 'Puma', 'CASPASE3', 'CASPASE7CLEAVEDD198', 'CASPASE8', 
        'Caspase-8-cleaved', 'PARP1', 'PARPCLEAVED', 'XIAP', 'SMAC'
    ]

    # Add any remaining proteins into a 'catch-all' structural/other kernel
}



# 2. Extract, Compute, and Normalize the RPPA Kernels
rppa_kernels = {}

for pathway_name, protein_list in rppa_pathways.items():
    # Only select proteins that actually exist in your dataframe columns
    valid_proteins = [p for p in protein_list if p in df_rppa.columns]
    
    # Skip if the pathway ends up empty
    if len(valid_proteins) == 0:
        continue
        
    # Extract the N x K subset for these specific proteins
    X_subset = df_rppa[valid_proteins].values
    
    # Compute the Gaussian (RBF) Kernel for continuous protein expression
    K_raw = rbf_kernel(X_subset)
    
    # Normalize the kernel (using the function we defined previously)
    K_norm = normalize_kernel(K_raw)
    
    # Add to the temporary dictionary
    rppa_kernels[pathway_name] = K_norm
    print(f"Computed {pathway_name} with {len(valid_proteins)} proteins.")

# 3. Update the Master Dictionary
# First, remove the old 'RPPA_Global' kernel so it doesn't compete with the new specific ones
if 'RPPA_Global' in normalized_computed_kernels:
    del normalized_computed_kernels['RPPA_Global']

# Add the new targeted protein kernels
normalized_computed_kernels.update(rppa_kernels)