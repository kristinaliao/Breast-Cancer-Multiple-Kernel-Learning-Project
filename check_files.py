# import os

# def list_files_os(directory_path='.'):
#     """
#     Lists all file names in the specified directory (non-recursive).
#     """
#     # Get all entries in the directory
#     entries = os.listdir(directory_path)
#     files = []
#     with open("data_case_IDs_rppa.txt", "w") as file:
#         for entry in entries:
#             # Construct the full path to check if it is a file
#             full_path = os.path.join(directory_path, entry)
#             if os.path.isfile(full_path):
#                 file.write(f"{entry}\n")

# # Example usage:
# directory_to_scan = './cleaned_RPPA' # Scans the current directory
# list_files_os(directory_to_scan)

from collections import Counter

with open('data_case_IDs.txt', 'r') as file:
    data_case_IDs = [line.strip() for line in file]

with open('data_case_IDs_rppa.txt', 'r') as file:
    data_case_IDs_rppa = [line.strip() for line in file]

with open('ILC_caseIDs_79.txt', 'r') as file:
    ILC_caseIDs_79 = [line.strip() for line in file]

if Counter(data_case_IDs) == Counter(ILC_caseIDs_79):
    print('equal')
