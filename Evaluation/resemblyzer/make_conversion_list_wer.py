# from utils.dataset import get_src_and_ref_mels, normalize, de_normalize
from utils.path import get_path, create_dir
from config import Arguments as args
import os
import random
import json

# 디렉토리 내의 파일 리스트 생성
def fList(directory_path):
        file_list = []
        # 주어진 디렉토리에서 파일 목록을 가져옴
        for root, dirs, files in os.walk(directory_path):
                for file in files:
                        file_list.append(os.path.join(root, file))
        return file_list

dataset_root = args.formatted_dir

unseen_speaker_list = ["p226", "p256", "p266", "p297", "p323", "p376"]

file_list = []

# 주어진 디렉토리에서 파일 목록을 가져옴
for root, dirs, files in os.walk(dataset_root):
    for file in files:
        file_list.append(os.path.join(root, file))

# 파일 리스트 중 speaker list 안에서 모두 겹치는 speech index 추출
files_num = []
for unseen in unseen_speaker_list:
    files_num.append(set([path.split('_')[1].split('.')[0] for path in file_list if unseen in path]))

common_elements = sorted(list(files_num[0].intersection(*files_num)))

seed_value = 42
random.seed(seed_value)
ref_random_integers = random.sample(common_elements[11:], 100)
src_random_integers = random.sample(common_elements[11:], 100)


gt_path_list = fList(args.dataset_path+'/txt/p226')
txt_path = args.dataset_path+'/txt/p226'
#100개
GT_txt = [get_path(txt_path, "p226_"+gt_i+".txt") for gt_i in src_random_integers]

for unseen_spk in unseen_speaker_list:
    

    
    
    
    #100개
    real_paths = [get_path(dataset_root, unseen_spk+"_"+rand_i+"_mic1.flac") for rand_i in src_random_integers]
    
    src_spk = [x for x in unseen_speaker_list if x != unseen_spk]*20
    
    ref_paths = [get_path(dataset_root, unseen_spk+"_"+rand_i+"_mic1.flac") for rand_i in ref_random_integers]
    
    
    src_paths = [get_path(dataset_root, rand_spk+"_"+src_rand_i+"_mic1.flac") for rand_spk, src_rand_i in zip(src_spk, src_random_integers)]


    print(unseen_spk)
    for src_path, ref_path in zip(src_paths, ref_paths):
        if not os.path.isfile(src_path) :
            print("[ERROR] No paths exist! Check your filename.: \n\t src_path: {}".format(src_path))
        elif not os.path.isfile(ref_path):
            print("[ERROR] No paths exist! Check your filename.: \n\t ref_path: {}".format(ref_path))
        else:
            pass
        
    abstracts_dict_list = {'gt': real_paths, 'ref_paths': ref_paths, 'src_paths': src_paths, 'gt_txt': GT_txt}
        
    # print(len(abstracts_dict_list))

    conversion_folder_path = '/root/sim/VoiceConversion/DL_Projects/conversion_files/unseen/'
    with open(conversion_folder_path+unseen_spk+'.json', 'w') as file:
        json.dump(abstracts_dict_list, file, indent=4)