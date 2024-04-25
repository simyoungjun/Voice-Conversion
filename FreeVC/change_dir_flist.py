import os
import argparse
from tqdm import tqdm
from random import shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="/home/sim/VoiceConversion/FreeVC/filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="/home/sim/VoiceConversion/FreeVC/filelists/val.txt", help="path to val list")
    parser.add_argument("--test_list", type=str, default="/home/sim/VoiceConversion/FreeVC/filelists/test.txt", help="path to test list")
    parser.add_argument("--source_dir", type=str, default="/shared/racoon_fast/sim/VCTK/preprocessed/vctk-16k", help="path to source dir")
    args = parser.parse_args()
    
            
    with open(args.train_list, encoding='utf-8') as f:
        split="|"
        filepaths_and_text = [line.strip().split(split) for line in f]
        print(filepaths_and_text)
        
    print("Writing", args.train_list)
    with open(args.train_list, "w") as f:
            for filepath in filepaths_and_text:
                wavpath = filepath[0].replace('DUMMY', '/shared/racoon_fast/sim/VCTK/preprocessed/vctk-16k')
                f.write(wavpath + "\n")
    
    
    with open(args.val_list, encoding='utf-8') as f:
        split="|"
        filepaths_and_text = [line.strip().split(split) for line in f]
    
    print("Writing", args.val_list)
    with open(args.val_list, "w") as f:
            for filepath in filepaths_and_text:
                wavpath = filepath[0].replace('DUMMY', '/shared/racoon_fast/sim/VCTK/preprocessed/vctk-16k')
                f.write(wavpath + "\n")
                
                
    with open(args.test_list, encoding='utf-8') as f:
        split="|"
        filepaths_and_text = [line.strip().split(split) for line in f]
    
    print("Writing", args.test_list)
    with open(args.test_list, "w") as f:
            for filepath in filepaths_and_text:
                wavpath = filepath[0].replace('DUMMY', '/shared/racoon_fast/sim/VCTK/preprocessed/vctk-16k')
                f.write(wavpath + "\n")

            