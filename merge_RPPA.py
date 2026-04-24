import pandas as pd
import os

data_folder = './cleaned_RPPA'

feature_column = 'peptide_target'
value_column = 'protein_expression'

all_patient_data = []

files = [f for f in os.listdir(data_folder)]
print(f"Found {len(files)} files. Starting merge...")

for filename in files:
    case_id = os.path.splitext(filename)[0]

    file_path = os.path.join(data_folder, filename)
    df = pd.read_csv(file_path, sep='\t')

    df = df.dropna(subset=[feature_column, value_column])

    patient_series = df.set_index(feature_column)[value_column]
    patient_series.name = case_id

    all_patient_data.append(patient_series)

rppa_matrix = pd.concat(all_patient_data, axis=1)
print(rppa_matrix.shape)
rppa_matrix = rppa_matrix.groupby([feature_column]).mean()
print(rppa_matrix.shape)

rppa_matrix = rppa_matrix.T
print(rppa_matrix.shape)

rppa_matrix.to_csv('merged_RPPA.csv')