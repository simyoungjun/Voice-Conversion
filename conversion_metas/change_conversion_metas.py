import os
import random
import json

# unseen_speaker_list = ["p226", "p256", "p266", "p297", "p323", "p376"]
file_num = 100
# conversion_path_dir = '/home/jinsung/Voice-Conversion/conversion_metas/VCTK_seen_pairs.txt'
# for spk in unseen_speaker_list:
with open(/home/jinsung/Voice-Conversion/conversion_metas/VCTK_seen_pairs.txt, 'r') as json_file:
    conversion_paths = json.load(json_file)

ref_paths = conversion_paths['ref_paths'][:file_num]
src_paths = conversion_paths['src_paths'][:file_num]


substring_to_replace = 'vctk-16'
replacement_string = 'wavlm


ref_paths_new = [ref_path.replace(substring_to_replace, replacement_string) for ref_path in ref_paths]
src_paths_new = [src_path.replace(substring_to_replace, replacement_string) for src_path in src_paths]
