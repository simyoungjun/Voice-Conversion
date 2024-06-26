from jiwer import wer, cer
import torchaudio
from transformers import HubertForCTC, Wav2Vec2Processor
import os
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import torch
import nltk
nltk.download('word_tokenize')
nltk.download('edit_distance')
nltk.download('punkt')

from evaluate import load
wer_class = load("wer")


import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis strings.
    """
    ref_words = nltk.word_tokenize(reference.lower())
    hyp_words = nltk.word_tokenize(hypothesis.lower())

    # Add dummy words to the beginning of the lists to align lengths
    ref_words = [''] + ref_words
    hyp_words = [''] + hyp_words

    # Initialize dynamic programming matrix
    dp = [[0] * (len(hyp_words)) for _ in range(len(ref_words))]

    for i in range(len(ref_words)):
        for j in range(len(hyp_words)):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                if ref_words[i] == hyp_words[j]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[-1][-1] / len(ref_words)


def cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER) between reference and hypothesis strings.
    """
    ref_length = len(reference)
    hyp_length = len(hypothesis)

    # Initialize dynamic programming matrix
    dp = [[0] * (hyp_length + 1) for _ in range(ref_length + 1)]

    for i in range(ref_length + 1):
        for j in range(hyp_length + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                if reference[i-1] == hypothesis[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[-1][-1] / ref_length

# Load the audio file
def get_wer(model, processor, gt_txt_fpaths, wav_fpaths, compare=True):
    wer_scores = []
    cer_scores = []
    comparing  = {"path":[], "wer": [], "cer": [], "GT": [], "pred": [], "model":[]}
    # compare = True
    
    for i in range(len(wav_fpaths)):
        #Ground truth text
        with open(gt_txt_fpaths[i], 'r') as file:
            ground_truth_transcription = file.read()
        
        ground_truth_transcription = re.sub(chars_to_ignore_regex, '', ground_truth_transcription)
        
        ground_truth_transcription = ground_truth_transcription.replace("\n", "").upper()
        
        hat_waveform, hat_sample_rate = torchaudio.load(wav_fpaths[i])

        hat_input_values = processor(hat_waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values


        # Get the embeddings from the model
        with torch.no_grad():
            hat_input_values = hat_input_values.to(1)
            model = model
            mel_logits = model(hat_input_values).logits

            predicted_ids = torch.argmax(mel_logits, dim=-1)
            predicted_transcription = processor.decode(predicted_ids[0]).upper()
        predicted_transcription = re.sub(chars_to_ignore_regex, '',predicted_transcription)
        
        wer_result = wer(ground_truth_transcription, predicted_transcription)
        cer_result = cer(ground_truth_transcription, predicted_transcription)        

        # wer_result = wer_class.compute(predictions=[predicted_transcription.upper()], references=[ground_truth_transcription])
        
        # # WER 계산

        wer_scores.append(wer_result)
        cer_scores.append(cer_result)
        if compare == True:
            comparing['path'].append(wav_fpaths[i].split('/')[-2])
            comparing['wer'].append(np.round(wer_result*100, 2))
            comparing['cer'].append(np.round(cer_result*100, 2))
            comparing['GT'].append(ground_truth_transcription)
            comparing['pred'].append(predicted_transcription)
        # Wav 별 WER 점수 확인
        print(f'{i}: {wer_result}')
        print(ground_truth_transcription)
        print(predicted_transcription)
        print('\n')
    
    return wer_scores, cer_scores, comparing



# ASR
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(1)


# ground_truth_transcription = "get_text_from_file(mel_filepath)"
# gt_path_list = fList(args.dataset_path+'/txt/p226')


models_paths = [
    # "/home/sim/VoiceConversion/FreeVC/output/freevc/VCTK_seen(1000)"
    # "/shared/racoon_fast/sim/results/V9_VQ1024_res_slice_cond_2/output/VCTK_seen_135(1000)_no_trim",
    # '/home/sim/VoiceConversion/FreeVC/output/freevc/LibriTTS_unseen_43seed(1000)',
    # '/shared/racoon_fast/sim/results/V8_VQ2048_SrcRef/output/LibriTTS_unseen_100(1000)'
    # '/shared/racoon_fast/sim/results/V8_VQ2048_SrcRef/output/LibriTTS_unseen_300(1000)',
    # '/shared/racoon_fast/sim/results/FreeVC/output/freevc/VCTK_seen(1000)',
    # '/shared/racoon_fast/sim/results/FreeVC/output/freevc/VCTK_seen(1000)',
    # '/shared/racoon_fast/sim/results/FreeVC/output/freevc/LibriTTS_unseen(1000)',
    
    "/shared/racoon_fast/sim/results/V9_VQ1024_res_slice_cond/output/VCTK_seen_134(1000)_no_trim"
    # '/shared/racoon_fast/sim/results/V9_VQ1024_res_slice_concat/output/VCTK_seen_28(1000)',
    # '/shared/racoon_fast/sim/results/V9_VQ1024_res_slice_concat/output/LibriTTS_unseen_28(1000)'
    
    # '/shared/racoon_fast/sim/results/V9_VQ1024_res_cond/output/LibriTTS_unseen_129(1000)'
    
    # "/shared/racoon_fast/sim/results/FreeVC/output/freevc/LibriTTS_unseen(1000)",
    # '/shared/racoon_fast/sim/results/YourTTS/output/LibriTTS_unseen_0(1000)',
    
    # "/shared/racoon_fast/sim/results/V8_VQ1024/output/LibriTTS_unseen_135_43seed(1000)",

                ]

# models_paths = [
#     # # '/shared/racoon_fast/sim/results/FreeVC/output/freevc/LibriTTS_unseen(1000)',
#     '/shared/racoon_fast/sim/results/FreeVC/output/freevc/VCTK_seen(1000)',
#     # # '/shared/racoon_fast/sim/results/V8/output/LibriTTS_unseen_57(1000)',
#     # '/shared/racoon_fast/sim/results/V8/output/VCTK_seen_57(1000)',
#     # # '/shared/racoon_fast/sim/results/VQMIVC/output/LibriTTS_unseen_0(1000)',
#     # '/shared/racoon_fast/sim/results/VQMIVC/output/VCTK_seen_0(1000)',
#     # # '/shared/racoon_fast/sim/results/YourTTS/output/LibriTTS_unseen_0(1000)',
#     # '/shared/racoon_fast/sim/results/YourTTS/output/VCTK_seen_0(1000)',
#     # # '/shared/racoon_fast/sim/results/V8_VQ1024/output/LibriTTS_unseen_90(1000)',
#     # # "/shared/racoon_fast/sim/results/V8_VQ256_no_affine_cond/output/LibriTTS_unseen_90(1000)",
#     # '/shared/racoon_fast/sim/results/V8_VQ1024_affine512/output/VCTK_seen_94(1000)',
#     # '/shared/racoon_fast/sim/results/V8_VQ1024/output/VCTK_seen_189(1000)',
#     # '/shared/racoon_fast/sim/results/V8_VQ1024_random_init/output/VCTK_seen_173(1000)',
#     # '/shared/racoon_fast/sim/results/V8_VQ1024/output/VCTK_seen_254(1000)',
#     "/shared/racoon_fast/sim/results/V8_VQ1024/output/VCTK_seen_189(1000)",
    
# ]
model_wav_list = []
names = []
for model_path in models_paths:
    names.append(model_path.split('/')[5])
    tgt_list = []
    cvt_list = []
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if "S" in file:
                tgt_list.append(os.path.join(root, file))
                # cvt_list.append(os.path.join(root, file))
                
                
            elif "C" in file:
                cvt_list.append(os.path.join(root, file))
    
    model_wav_list.append([tgt_list, cvt_list])
    print(len(tgt_list))


def txt_fpath_from_wav(wav_fpath):
    if "LibriTTS" in wav_fpath:
        # txt_fpath = wav_fpath.replace('.wav', '')
        txt_dir = '/shared/racoon_fast/sim/LibriTTS/train-clean-100'
        tmp = wav_fpath.split('!')[-1].split('_')
        txt_fpath = os.path.join(txt_dir, tmp[0], tmp[1], wav_fpath.split('!')[-1])
        txt_fpath = txt_fpath.replace('.wav', '.original.txt')
    elif "VCTK" in wav_fpath:
        txt_dir = '/shared/racoon_fast/sim/VCTK/txt'
        txt_fpath = os.path.join(txt_dir, wav_fpath.split('!')[-1].split('_')[0], wav_fpath.split('!')[-1])
        txt_fpath = txt_fpath.replace('.wav', '.txt')
    return txt_fpath

model_fpaths_list = []
for tgt_paths, cvt_paths in model_wav_list:	 
    tgt_txt_paths = [txt_fpath_from_wav(wav_fpath) for wav_fpath in tgt_paths]
    # cvt_txt_paths = [txt_fpath_from_wav(wav_fpath) for wav_fpath in tgt_paths]
    model_fpaths_list.append([tgt_txt_paths, cvt_paths])

total_scores = []
total_max = []
total_min = []
total_score = []
total_comparing = []
model_names = []
for (tgt_txt_paths, cvt_wav_paths) in tqdm(model_fpaths_list):
    model_names.append(cvt_wav_paths[0].split('/output')[0].split('/')[-1])
    wer_scores, cer_scores, comparing = get_wer(model, processor, tgt_txt_paths, cvt_wav_paths)
    total_scores.append([wer_scores, cer_scores])
    total_comparing.append(comparing)

# wer_cer_comapring.txt 에 비교할 수 있는 결과 저장
comparing=False
models_num = len(total_comparing)
if models_num == 2 and comparing==True:
    compareing_write = {'wer': [total_comparing[0]['wer'], total_comparing[1]['wer']],
                        'cer': [total_comparing[0]['cer'], total_comparing[1]['cer']],
                        'path': total_comparing[0]['path'],
                        'GT': total_comparing[0]['GT'],
                        'pred': [total_comparing[0]['pred'], total_comparing[1]['pred']],
                        'model': [model_names[0], model_names[1]]
                        }

    out_comparing_file = 'wer_cer_comparing_VCTK.txt'
    with open(out_comparing_file, 'w') as txt_file:
        for i in range(len(total_comparing[0]['wer'])):
            print('9')
            txt_file.write(f"PATH:{compareing_write['path'][i]}\n")
            txt_file.write(f"WER:{[compareing_write['wer'][0][i], compareing_write['wer'][1][i]]} | CER: {[compareing_write['cer'][0][i], compareing_write['cer'][1][i]]}\n")
            txt_file.write(f"Transcipt: GT|{compareing_write['model'][0]}|{compareing_write['model'][1]}\n")
            txt_file.write(f"{compareing_write['GT'][i]}\n{compareing_write['pred'][0][i]}\n{compareing_write['pred'][1][i]}\n\n")
        

# scores 비교
total_scores = np.array(total_scores)
total_avg = np.mean(total_scores, axis=2)

if total_scores.shape[0] > 1:
    for i in range(total_scores.shape[2]):
        print(f'WER {i}: {round(total_scores[0, 0, i], 4)}, {round(total_scores[1, 0, i], 4)}')
        print(f'CER {i}: {round(total_scores[0, 1, i], 4)}, {round(total_scores[1, 1, i], 4)}')
        print('\n')


print(total_avg)

