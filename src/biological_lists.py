# Leakage kernels to be removed (cell-cycle/proliferation proxy kernels)
LEAKAGE_KERNELS = [
    'HALLMARK_SPERMATOGENESIS', 
    'HALLMARK_G2M_CHECKPOINT', 
    'HALLMARK_E2F_TARGETS', 
    'HALLMARK_MITOTIC_SPINDLE',
    'HALLMARK_DNA_REPAIR'
]

# Custom biological gene lists for ILC
ADHESION_GENES = ['CDH1', 'CTNNA1', 'CTNNB1', 'CTNND1']
AKT_GENES = ['PTEN', 'PIK3CA', 'AKT1', 'AKT2', 'AKT3', 'INPP4B', 'EGFR', 'ERBB2', 'STAT3']
TF_GENES = ['FOXA1', 'GATA3', 'RUNX1', 'TBX3', 'ESR1']

# Dictionary of custom lists for mRNA kernels
CUSTOM_LISTS = {
    'ILC_Adhesion': ADHESION_GENES,
    'ILC_AKT_Pathway': AKT_GENES,
    'ILC_TF_Drivers': TF_GENES
}

# RPPA functional modules based on exact column names
RPPA_PATHWAYS = {
    # 1. PI3K/AKT/mTOR Survival & Metabolism
    'RPPA_PI3K_AKT_mTOR': [
        'AKT', 'AKT_pS473', 'AKT_pT308', 'Akt2', 'Akt2_pS474', 'PTEN', 'INPP4B', 
        'PI3KP110ALPHA', 'PI3KP85', 'PI3K-p110-b', 'MTOR', 'MTOR_pS2448', 
        'TUBERIN', 'TUBERIN_pT1462', 'TSC1', 'P70S6K1', 'P70S6K_pT389', 
        'S6', 'S6_pS235S236', 'S6_pS240S244', '4EBP1', '4EBP1_pS65', 
        '4EBP1_pT37T46', '4EBP1_pT70', 'PRAS40', 'PRAS40_pT246'
    ],

    # 2. Hormone Receptors & Receptor Tyrosine Kinases (RTKs)
    'RPPA_Receptors_RTK': [
        'ERALPHA', 'ERALPHA_pS118', 'PR', 'AR', 'EGFR', 'EGFR_pY1068', 
        'EGFR_pY1173', 'HER2', 'HER2_pY1248', 'HER3', 'HER3_pY1289', 
        'IGFRb', 'IGF1R_pY1135Y1136', 'CMET', 'CMET_pY1235'
    ],

    # 3. Adhesion, EMT & Structure
    'RPPA_Adhesion_EMT': [
        'ECADHERIN', 'NCADHERIN', 'PCADHERIN', 'BETACATENIN', 
        'b-Catenin_pT41_S45', 'FIBRONECTIN', 'FN14', 'SNAIL', 'ZEB1', 
        'CLAUDIN7', 'COLLAGENVI', 'CAVEOLIN1'
    ],

    # 4. MAPK/ERK Proliferation Cascade
    'RPPA_MAPK_ERK': [
        'ARAF', 'ARAF_pS299', 'BRAF', 'BRAF_pS445', 'CRAF', 'CRAF_pS338', 
        'MEK1', 'MEK1_pS217S221', 'MEK2', 'MAPK_pT202Y204', 'p44-42-MAPK', 
        'P38MAPK', 'P38_pT180Y182', 'p38-a', 'JNK2', 'JNK_pT183Y185'
    ],

    # # 5. Cell Cycle & DNA Repair
    # 'RPPA_CellCycle_Repair': [
    #     'CDK1', 'CDK1_pT14', 'CDK1_pY15', 'CYCLINB1', 'CYCLIND1', 'CYCLINE1', 
    #     'CYCLINE2', 'P16INK4A', 'P21', 'P27', 'P27_pT157', 'P27_pT198', 
    #     'RB', 'RB_pS807S811', 'P53', 'ATM', 'ATM_pS1981', 'ATR', 'ATR_pS428', 
    #     'CHK1', 'CHK1_pS296', 'CHK1_pS345', 'CHK2', 'CHK2_pT68', 
    #     'H2AX_pS139', 'H2AX_pS140'
    # ],

    # # 6. Apoptosis & Survival
    # 'RPPA_Apoptosis': [
    #     'BAX', 'BAK', 'BAD_pS112', 'BCL2', 'BCL2A1', 'BCLXL', 'BIM', 'BID', 
    #     'Mcl-1', 'Puma', 'CASPASE3', 'CASPASE7CLEAVEDD198', 'CASPASE8', 
    #     'Caspase-8-cleaved', 'PARP1', 'PARPCLEAVED', 'XIAP', 'SMAC'
    # ]
}


