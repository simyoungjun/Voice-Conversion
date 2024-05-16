import shutil
import os

def move_folder(source, destination):
    # Check if the destination directory exists, if not, create it
    if not os.path.exists(destination):
        os.makedirs(destination)
    # Move the folder
    shutil.move(source, destination)
    print(f"Folder moved from {source} to {destination}")

# Define the source and destination paths
source_path = []
source_path.append('/home/sim/VoiceConversion/V8/output/LibriTTS_unseen_57(1000)')
source_path.append('/home/sim/VoiceConversion/V8/output/VCTK_seen_57(1000)')

source_path.append('/home/sim/VoiceConversion/FreeVC/output/freevc/LibriTTS_unseen(1000)')
source_path.append('/home/sim/VoiceConversion/FreeVC/output/freevc/VCTK_seen(1000)')

source_path.append('/home/sim/VoiceConversion/YourTTS/output/LibriTTS_unseen_0(1000)')
source_path.append('/home/sim/VoiceConversion/YourTTS/output/VCTK_seen_0(1000)')

source_path.append('/home/sim/VoiceConversion/VQMIVC/output/VCTK_seen_0(1000)')

# destination_path = '/path/to/destination/folder'

for src_path in source_path:
    # folder_name = src_path.split('/')[4]
    destination_path = src_path.replace('/home/sim/VoiceConversion','/shared/racoon_fast/sim/results' )
    # Call the function
    if os.path.exists(destination_path):
        print(f"Error: Destination path '{destination_path}' already exists.")
    # Copy the folder
    shutil.copytree(src_path, destination_path)
    print(f"Folder copied from {src_path} to {destination_path}")