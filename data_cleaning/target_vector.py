import pandas as pd

target_vector = pd.read_csv('ILC_79_clinical.csv')
target_vector = target_vector[['Case.ID', 'ProliferationScore']]
print(target_vector.shape)

target_vector.to_csv('target_vector.csv')