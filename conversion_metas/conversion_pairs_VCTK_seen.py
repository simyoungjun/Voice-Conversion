#%

import os
import random
import json
import numpy as np
from tqdm import tqdm

random.seed(42)

def get_path(*args):
        return os.path.join('', *args)

# dataset_root = '/root/sim/VoiceConversion/Datasets/VCTK/VCTK-Corpus/txt/'
wavs_root = '/shared/racoon_fast/sim/VCTK/preprocessed/vctk-16k'

unseen = ['p226', 'p256', 'p266', 'p297', 'p323', 'p376']


spk_list = []
text_path_dict = {}
# 주어진 디렉토리에서 spk 목록 가져옴
for spk in os.listdir(wavs_root):
    if spk not in unseen:
        spk_list.append(os.path.join(wavs_root, spk))

output_file = '/home/sim/VoiceConversion/conversion_metas/VCTK_seen_pairs(1000).txt'




# with open(output_file, 'w') as txt_file:
# 같은 문장이 존재하는 spk 찾기
with open(output_file, 'w') as txt_file:
    for i in range(1000):
        src_spk_dir, tgt_spk_dir = random.sample(spk_list, 2) 
        
        src_speech = random.sample([file for file in os.listdir(src_spk_dir) if file.endswith(".wav")], 1)
        tgt_speech = random.sample([file for file in os.listdir(tgt_spk_dir) if file.endswith(".wav")], 1)
        
        tgt_path = os.path.join(src_spk_dir, src_speech[0])
        src_path = os.path.join(tgt_spk_dir, tgt_speech[0])
        title = 'title'
        txt_file.write(f'{title}|{tgt_path}|{src_path}\n')

# print('Done')
