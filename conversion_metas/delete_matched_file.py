import os
import glob

def delete_matched_files(directory, pattern):
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(directory):
        # Construct the full file path
        for file in files:
            if pattern in file:
                full_path = os.path.join(root, file)
                # Delete the file
                os.remove(full_path)
                print(f"Deleted {full_path}")

# Specify the directory and pattern to search
# directory_path = '/shared/racoon_fast/sim/VCTK/preprocessed' # 이 폴더에서
directory_path = '/shared/racoon_fast/sim/LibriTTS/preprocessed' # 이 폴더에서

pattern = '_reference_gen' # 이 문자열 들어간 하위 파일 전부 지움
# pattern = '_source_gen' # 이 문자열 들어간 하위 파일 전부 지움

# Call the function
delete_matched_files(directory_path, pattern)