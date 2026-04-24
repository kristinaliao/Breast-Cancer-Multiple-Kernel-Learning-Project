import pandas as pd
import os

data_folder = './cleaned_mRNA'

gene_column = 'gene_name'
value_column = 'tpm_unstranded'

all_patient_data = []

files = [f for f in os.listdir(data_folder)]
print(f"Found {len(files)} files. Starting merge...")

for filename in files:
    case_id = os.path.splitext(filename)[0]

    file_path = os.path.join(data_folder, filename)
    df = pd.read_csv(file_path, sep='\t', comment='#')

    df = df.dropna(subset=[gene_column])

    patient_series = df.set_index(gene_column)[value_column]
    patient_series.name = case_id

    all_patient_data.append(patient_series)

mrna_matrix = pd.concat(all_patient_data, axis=1)
print(mrna_matrix.shape)
mrna_matrix = mrna_matrix.groupby([gene_column]).mean()
print(mrna_matrix.shape)

mrna_matrix = mrna_matrix.T
print(mrna_matrix.shape)

mrna_matrix.to_csv('merged_mRNA_TPM.csv')