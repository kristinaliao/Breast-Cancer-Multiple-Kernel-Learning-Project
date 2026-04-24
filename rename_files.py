import os
import pandas as pd
import shutil

sample_sheet_path = 'ILC_RPPA_sample_sheet.tsv'

download_dir = './ILC_RPPA'

output_dir = './cleaned_RPPA'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ss_df = pd.read_csv(sample_sheet_path, sep='\t')

print(f"Loaded sample sheet with {len(ss_df)} entries.")

success_count = 0
not_found_count = 0

for index, row in ss_df.iterrows():
    file_name = row['File Name']
    case_id = row['Case ID']
    
    file_id = row['File ID']
    source_path = os.path.join(download_dir, file_id, file_name)

    if os.path.exists(source_path):
        destination_path = os.path.join(output_dir, case_id)
        shutil.copy2(source_path, destination_path)
        success_count += 1
    else:
        print(f"Warning: File {file_name} not found in {download_dir}")
        not_found_count += 1

print(f"\nProcessing Complete!")
print(f"Files successfully organized: {success_count}")
print(f"Files not found: {not_found_count}")
print(f"All files are now in: {os.path.abspath(output_dir)}")
