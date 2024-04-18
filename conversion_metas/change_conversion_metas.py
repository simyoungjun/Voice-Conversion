import os
import random
import json

unseen_speaker_list = ["p226", "p256", "p266", "p297", "p323", "p376"]
file_num = 100
conversion_path_dir = '/root/sim/VoiceConversion/VQVC/conversion_metas/unseen/'
for spk in unseen_speaker_list:
    with open(conversion_path_dir+spk+'.json', 'r') as json_file:
        conversion_paths = json.load(json_file)

    ref_paths = conversion_paths['ref_paths'][:file_num]
    src_paths = conversion_paths['src_paths'][:file_num]
    
    
    substring_to_replace = 'Datasets/VCTK/VCTK-Corpus/formatted'
    replacement_string = 'FreeVC/dataset/vctk-16k/'+ spk
    
    
    ref_paths_new = [ref_path.replace(substring_to_replace, replacement_string) for ref_path in ref_paths]
    src_paths_new = [src_path.replace(substring_to_replace, replacement_string) for src_path in src_paths]
