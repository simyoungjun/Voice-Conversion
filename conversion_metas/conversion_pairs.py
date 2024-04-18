#%

import os
import random
import json
import numpy as np
from tqdm import tqdm


def get_path(*args):
        return os.path.join('', *args)

dataset_root = '/shared/racoon_fast/sim/VCTK/txt'
wavs_root = '/shared/racoon_fast/sim/VCTK/preprocessed/vctk-16k'

file_list = []
text_path_dict = {}
# 주어진 디렉토리에서 파일 목록을 가져옴
for root, dirs, files in os.walk(dataset_root):
    file_list = []
    for file in files:
        file_list.append(os.path.join(root, file))
    text_path_dict[root.split('/')[-1]] = sorted(file_list)
    
        
text_dict = {}
for key, paths in text_path_dict.items():
    text_list = []
    for path in paths:
        with open(path, 'r') as text_file:
            # text_list.append(text_file.read())
            text_list.append(text_file.read().split('.')[0])
            
            # content = text_file.read()
    text_dict[key] = text_list
    
text_dict = dict(sorted(text_dict.items(), key=lambda item: item[0]))


#%

#
# Conversion file list
output_file = 'conversion_metas/unseen_pairs.txt'

# Write the indices to a text file using the built-in open function



#Unseen list
unseen = ['p226', 'p256', 'p266', 'p297', 'p323', 'p376']
unseen_values = [text_dict[key] for key in unseen if key in text_dict]
# Specify the output file path
output_file = '/home/sim/VoiceConversion/conversion_metas/unseen_pairs.txt'


# with open(output_file, 'w') as txt_file:
# 같은 문장이 존재하는 spk 찾기
with open(output_file, 'w') as txt_file:
    for tgt_spk_i, tgt_spk in enumerate(unseen): # target speaker
        indices_all = []
        for target_sentence in tqdm(text_dict[tgt_spk]): # target spk 가 말한 corpus
            indices = []
            for unseen_corpus in unseen_values: # unseen spk 들이 말한 corpus
                #같은 문장 있는지 찾음
                list_indices = [i for i, item in enumerate(unseen_corpus) if item == target_sentence] 
                # 없으면 -1 저장, 있으면 문장의 idx 저장
                indices.append(list_indices[0] if list_indices else -1)
            # 각 문장 별 반환값 저장
            indices_all.append(indices)

        pair_data = np.array(indices_all)
        #각 문장 별로 pair 가능한 spk 확인
        #tgt_sen_i: tgt corpus에서 해당 문장이 있는 인덱스
        for tgt_sen_i, sen in enumerate(pair_data): 
            count = 6 - np.count_nonzero(sen == -1) # 같은 문장이 존재하는지 개수
            mask = sen != -1
            spk_indices = np.where(mask)[0] # -1아닌 요소 spk idx 반환
            possible_pair = spk_indices
            # 문장 중복되는 spk 있으면
            other_unseen = np.where(unseen != tgt_spk_i)
            if count > 1:
                
                src = [[unseen[src_spk], sen[src_spk]] for src_spk in possible_pair if src_spk != tgt_spk_i]
                print(tgt_sen_i)
                print(sen)
                print(f'tgt: {tgt_spk}_all, src: {src}, GT: {tgt_spk}_{sen[tgt_spk_i]}')
                print(f'{text_dict[src[0][0]][src[0][1]]}')
                print(f'{text_dict[tgt_spk][sen[tgt_spk_i]]}')
                print()
                
                for i in range(len(src)):
                    random_number = random.randint(1, len(pair_data)-1)
                    tgt_path = text_path_dict[tgt_spk][random_number].replace(dataset_root, wavs_root).replace('txt', 'wav', -1)
                    src_path = text_path_dict[src[i][0]][src[i][1]].replace(dataset_root, wavs_root).replace('txt', 'wav', -1)
                    gt_path = f'{wavs_root}{tgt_spk}/{tgt_spk}_{src_path.split("_")[-1].split(".")[0]}.wav'
                    
                    # tgt_path = f'{wavs_root}{tgt_spk}/{tgt_spk}_{random_number}.wav'
                    # src_path = f'{wavs_root}{src[i][0]}/{src[i][0]}_{src[i][1]}.wav'
                    # gt_path = f'{wavs_root}{tgt_spk}/{tgt_spk}_{src_path.split("_")[-1].split(".")[0]}.wav'
                    txt_file.write(f'{gt_path}|{tgt_path}|{src_path}\n')
                
            
            # print(target_element)
            # print(indices)
            # print(count)
            # print(f'possible pair: {np.array(unseen)[spk_indices]}\n {indices_arr.reshape(-1)}\n')

#%

    

#%

# # 파일 리스트 중 speaker list 안에서 모두 겹치는 speech index 추출
# files_num = []
# for unseen in unseen_speaker_list:
#     files_num.append(set([path.split('_')[1].split('.')[0] for path in file_list if unseen in path]))

# common_elements = sorted(list(files_num[0].intersection(*files_num)))


# seed_value = 42
# random.seed(seed_value)
# ref_random_integers = random.sample(common_elements[11:], 100)
# src_random_integers = random.sample(common_elements[11:], 100)

# for unseen_spk in unseen_speaker_list:
#     #10개
#     GT_speech = [get_path(dataset_root, unseen_spk+'/'+unseen_spk+"_"+f'{gt_i:03}'+".wav") for gt_i in range(1, 11)]
#     #100개
#     real_paths = [get_path(dataset_root, unseen_spk+'/'+unseen_spk+"_"+rand_i+".wav") for rand_i in src_random_integers]
    
#     src_spk = [x for x in unseen_speaker_list if x != unseen_spk]*20
    
#     ref_paths = [get_path(dataset_root, unseen_spk+'/'+unseen_spk+"_"+rand_i+".wav") for rand_i in ref_random_integers]
    
    
#     src_paths = [get_path(dataset_root, rand_spk+'/'+rand_spk+"_"+src_rand_i+".wav") for rand_spk, src_rand_i in zip(src_spk, src_random_integers)]


#     # result = [path.split('_')[1].split('.')[0] for path in ref_paths if 'p226' in path]
#     # print(result)
    
#     # print(unseen_spk)
#     # for src_path, ref_path in zip(src_paths, ref_paths):
#     #     if not os.path.isfile(src_path) :
#     #         print("[ERROR] No paths exist! Check your filename.: \n\t src_path: {}".format(src_path))
#     #     elif not os.path.isfile(ref_path):
#     #         print("[ERROR] No paths exist! Check your filename.: \n\t ref_path: {}".format(ref_path))
#     #     else:
#     #         pass
        
#     abstracts_dict_list = {'gt': GT_speech, 'real_paths': real_paths, 'ref_paths': ref_paths, 'src_paths': src_paths}
        
#     # print(len(abstracts_dict_list))

#     conversion_folder_path = '/root/sim/VoiceConversion/FreeVC/conversion_metas/unseen/'
#     with open(conversion_folder_path+unseen_spk+'.json', 'w') as file:
#         json.dump(abstracts_dict_list, file, indent=4)