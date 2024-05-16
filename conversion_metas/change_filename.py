import os

def rename_files_recursively(root_directory):
    # Walk through the directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            # Check if '.wavgen' is in the filename
            if 'C;' in filename:
                k = 'C;'
            elif 'S;' in filename:
                k = 'S;'            
            elif 'T;' in filename:
                k = 'T;'         
            else:
                k = None
            if k != None:
                # Create the new filename by replacing '.wavgen' with a space
                new_filename = filename.replace(k, k[0]+'!')
                
                # Full path for old and new filenames
                old_file_path = os.path.join(dirpath, filename)
                new_file_path = os.path.join(dirpath, new_filename)
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{old_file_path}' to '{new_file_path}'")

# Specify the path to the directory containing the files
directory_path = '/shared/racoon_fast/sim/results/VQMIVC/output'

# Call the function
rename_files_recursively(directory_path)