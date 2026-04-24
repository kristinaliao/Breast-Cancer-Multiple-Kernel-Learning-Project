import pandas as pd

file_path = './data_checking/ILC_caseIDs_79.txt'
with open(file_path, 'r') as f:
    caseIDs = [line.strip() for line in f]

clinical_raw = pd.read_csv('clinical_data.txt', sep="\t")
clinical_data = clinical_raw[clinical_raw['bcr_patient_barcode'].isin(caseIDs)]
print(clinical_data.shape)

supplementary_data = pd.read_csv('ILC_79_clinical.csv')
print(supplementary_data.shape)

clinical_data = clinical_data.rename(columns={'bcr_patient_barcode':'Case_ID'})
supplementary_data = supplementary_data.rename(columns={'Case.ID':'Case_ID'})

df_combined = pd.merge(clinical_data, supplementary_data, on='Case_ID')

feature_columns = ['Case_ID', 'TumorPurity', 'EMT score', 'age_at_diagnosis', 'menopause_status', 'ajcc_pathologic_tumor_stage', 'PAM50', 'er_status_by_ihc', 'pr_status_by_ihc', 'her2_status_by_ihc']

df_final = df_combined[feature_columns].set_index('Case_ID')

#Menopause status
meno_map = {'Post (prior bilateral ovariectomy OR >12 mo since LMP with no prior hysterectomy)':1,
            'Pre (<6 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)':0}
df_final['Menopause_binary'] = df_final['menopause_status'].map(meno_map)

#ajcc stage 
ajcc_map = {'Stage I':1,
            'Stage IA':1,
            'Stage II':2,
            'Stage IIA':2,
            'Stage IIB':2,
            'Stage III':3,
            'Stage IIIA':3,
            'Stage IIIB':3,
            'Stage IIIC':3
            }
df_final['Tumor_stage_numeric'] = df_final['ajcc_pathologic_tumor_stage'].map(ajcc_map)

#PAM50
df_final = pd.get_dummies(df_final, columns=['PAM50'], prefix=['Subtype'],dtype=int)

ihc_map = {'Positive': 1, 'Negative': 0}
df_final['ER_binary'] = df_final['er_status_by_ihc'].map(ihc_map)
df_final['PR_binary'] = df_final['pr_status_by_ihc'].map(ihc_map)
df_final['HER2_binary'] = df_final['her2_status_by_ihc'].map(ihc_map)

df_final = df_final.drop(columns=['er_status_by_ihc', 'pr_status_by_ihc', 'her2_status_by_ihc', 'menopause_status', 'ajcc_pathologic_tumor_stage'])

df_final['age_at_diagnosis'] = df_final['age_at_diagnosis'].astype(int)

df_final = df_final.fillna(df_final.median())

df_final.to_csv('cleaned_clinical_data.csv')

X_clinical = df_final.drop(columns=['TumorPurity', 'EMT score', 'ER_binary','PR_binary','HER2_binary'])
X_clinical.to_csv('X_clinical.csv')
