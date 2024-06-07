
#%
import whisper
import os
from tqdm import tqdm
import numpy as np
from evaluate import load
import re
import nltk
nltk.download('word_tokenize')
nltk.download('edit_distance')
nltk.download('punkt')
wer = load("wer")

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
models_paths = [
    # "/home/sim/VoiceConversion/FreeVC/output/freevc/VCTK_seen(1000)",
    "/home/sim/VoiceConversion/FreeVC/output/freevc/LibriTTS_unseen(1000)",
    '/shared/racoon_fast/sim/results/YourTTS/output/LibriTTS_unseen_0(1000)',
    "/shared/racoon_fast/sim/results/V8_VQ1024/output/LibriTTS_unseen_135(1000)",

                ]
models_paths = [
    # # '/shared/racoon_fast/sim/results/FreeVC/output/freevc/LibriTTS_unseen(1000)',
    # '/shared/racoon_fast/sim/results/FreeVC/output/freevc/VCTK_seen(1000)',
    # # '/shared/racoon_fast/sim/results/V8/output/LibriTTS_unseen_57(1000)',
    # '/shared/racoon_fast/sim/results/V8/output/VCTK_seen_57(1000)',
    # # '/shared/racoon_fast/sim/results/VQMIVC/output/LibriTTS_unseen_0(1000)',
    # '/shared/racoon_fast/sim/results/VQMIVC/output/VCTK_seen_0(1000)',
    # # '/shared/racoon_fast/sim/results/YourTTS/output/LibriTTS_unseen_0(1000)',
    # '/shared/racoon_fast/sim/results/YourTTS/output/VCTK_seen_0(1000)',
    # # '/shared/racoon_fast/sim/results/V8_VQ1024/output/LibriTTS_unseen_90(1000)',
    # # "/shared/racoon_fast/sim/results/V8_VQ256_no_affine_cond/output/LibriTTS_unseen_90(1000)",
    # '/shared/racoon_fast/sim/results/V8_VQ1024_affine512/output/VCTK_seen_94(1000)',
    # '/shared/racoon_fast/sim/results/V8_VQ1024/output/VCTK_seen_189(1000)',
    # '/shared/racoon_fast/sim/results/V8_VQ1024_random_init/output/VCTK_seen_173(1000)',
    # '/shared/racoon_fast/sim/results/V8_VQ1024/output/VCTK_seen_254(1000)',
]
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

model = whisper.load_model("base")
chars_to_ignore_regex = '[\,\.\?\!\-\;\:\"]'

for (tgt_txt_paths, cvt_wav_paths) in tqdm(model_fpaths_list):
    wer_scores = []
    cer_scores = []
    for i in range(len(cvt_wav_paths)):
    # wer_scores, cer_scores = get_wer(model, processor, tgt_txt_paths, cvt_wav_paths)

        with open(tgt_txt_paths[i], 'r') as file:
            ground_truth_transcription = file.read()
        
        ground_truth_transcription = re.sub(chars_to_ignore_regex, '', ground_truth_transcription).replace("\n", "")

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(cvt_wav_paths[i])
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        # _, probs = model.detect_language(mel)
        # print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions(language='en')
        result = whisper.decode(model, mel, options)

        ground_truth_transcription = ground_truth_transcription.upper()
        predicted_transcription = result.text.upper()
        predicted_transcription = re.sub(chars_to_ignore_regex, '',predicted_transcription)
        # print the recognized text
        wer_result = wer(ground_truth_transcription, predicted_transcription)
        cer_result = cer(ground_truth_transcription, predicted_transcription)   
        if wer_result >= 1:
            continue 
        wer_scores.append(wer_result)
        cer_scores.append(cer_result)
        print(f'{i}: {wer_result}, {cvt_wav_paths[i]}')
        print(ground_truth_transcription)
        print(predicted_transcription)
        print('\n')
        # wer_score = wer.compute(predictions=result, references=references)
        # print(wer_score)        # total_scores.append([wer_scores, cer_scores])
    total_scores.append([wer_scores, cer_scores])
    
    

total_scores = np.array(total_scores)
total_avg = np.mean(total_scores, axis=2)
print(total_avg)