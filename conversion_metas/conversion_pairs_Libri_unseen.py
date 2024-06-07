#%

import os
import random
import json
import numpy as np
from tqdm import tqdm

seed = 43
random.seed(seed)

def get_path(*args):
        return os.path.join('', *args)

# dataset_root = '/root/sim/VoiceConversion/Datasets/VCTK/VCTK-Corpus/txt/'
wavs_root = '/shared/racoon_fast/sim/LibriTTS/preprocessed/LibriTTS-16k'

spk_list = []
text_path_dict = {}
# 주어진 디렉토리에서 spk 목록 가져옴
for spk in os.listdir(wavs_root):
    spk_list.append(os.path.join(wavs_root, spk))

output_file = f'/home/sim/VoiceConversion/conversion_metas/LibriTTS_unseen_pairs(1000)_{seed}.txt'



# with open(output_file, 'w') as txt_file:
# 같은 문장이 존재하는 spk 찾기
with open(output_file, 'w') as txt_file:
    for i in range(1000):
        src_spk_dir, tgt_spk_dir = random.sample(spk_list, 2) 
        
        src_speech = random.sample(os.listdir(src_spk_dir), 1)
        tgt_speech = random.sample(os.listdir(tgt_spk_dir), 1)
        
        tgt_path = os.path.join(src_spk_dir, src_speech[0])
        src_path = os.path.join(tgt_spk_dir, tgt_speech[0])
        title = 'title'
        txt_file.write(f'{title}|{tgt_path}|{src_path}\n')

# print('Done')
