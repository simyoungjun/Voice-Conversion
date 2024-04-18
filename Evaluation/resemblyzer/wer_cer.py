from jiwer import wer
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
import os
import numpy as np
from config import Arguments as args
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import torch
import nltk



# 디렉토리 내의 파일 리스트 생성
def fList(directory_path):
        file_list = []
        # 주어진 디렉토리에서 파일 목록을 가져옴
        for root, dirs, files in os.walk(directory_path):
                for file in files:
                        file_list.append(os.path.join(root, file))
        return file_list
    
unseen_speaker_list = ["p226", "p256", "p266", "p297", "p323", "p376"]

        # Load the audio file
def get_wer(model, processor, gt_txt_fpaths, wav_fpaths):
    scores = []
    for i in range(len(wav_fpaths)):
        #Ground truth text
        with open(gt_txt_fpaths[i], 'r') as file:
            ground_truth_transcription = file.read()
        
        hat_waveform, hat_sample_rate = torchaudio.load(wav_fpaths[i])
        # Preprocess the audio file
        #input_values = processor(hat_waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        hat_input_values = processor(hat_waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values


        # Get the embeddings from the model
        with torch.no_grad():
            #input_values = input_values.to(0)
            hat_input_values = hat_input_values.to(0)
            model = model.to(0)
            #logits = model(input_values).logits
            mel_logits = model(hat_input_values).logits

            predicted_ids = torch.argmax(mel_logits, dim=-1)
            predicted_transcription = processor.batch_decode(predicted_ids)[0]

        # _wer_ = wer(ground_truth_transcription, predicted_transcription)
        model_output_words = nltk.word_tokenize(predicted_transcription.lower())
        ground_truth_words = nltk.word_tokenize(ground_truth_transcription.lower())
        edit_distance = nltk.edit_distance(model_output_words, ground_truth_words)
        # print(f"predicted_transcription: {model_output_words}")
        # print(f"GT_transcription: {ground_truth_words}")

        # WER 계산
        word_error_rate = edit_distance / len(ground_truth_words)
        scores.append(word_error_rate)
        
    return scores


# ASR
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
ground_truth_transcription = "get_text_from_file(mel_filepath)"

gt_path_list = fList(args.dataset_path+'/txt/p226')



total_scores = []
total_max = []
total_min = []
for spk in unseen_speaker_list:
        with open(args.conversion_folder_path+spk+'.json', 'r') as json_file:
                conversion_paths = json.load(json_file)
        

        
        gt_txt_fpaths = conversion_paths['gt_txt']
        
        real_wav_fpaths = conversion_paths['gt']
        
        VQ256_wav_fpaths = fList(args.converted_sample_root+'VQVC_256')
        VQ256_wav_fpaths = [path for path in tqdm(VQ256_wav_fpaths) if spk in path.split('_')[2]]
        
        VQ512_wav_fpaths = fList(args.converted_sample_root+'VQVC_512')
        VQ512_wav_fpaths = [path for path in tqdm(VQ512_wav_fpaths) if spk in path.split('_')[2]]
        
        VQ1024_wav_fpaths = fList(args.converted_sample_root+'VQVC_1024')
        VQ1024_wav_fpaths = [path for path in tqdm(VQ1024_wav_fpaths) if spk in path.split('_')[2]]
        
        CVQ256_wav_fpaths = fList(args.converted_sample_root+'model_CVQ_256')
        CVQ256_wav_fpaths = [path for path in tqdm(CVQ256_wav_fpaths) if spk in path.split('_')[3]]
        
        CVQ512_wav_fpaths = fList(args.converted_sample_root+'model_CVQ_512')
        CVQ512_wav_fpaths = [path for path in tqdm(CVQ512_wav_fpaths) if spk in path.split('_')[3]]
        
        CVQ1024_wav_fpaths = fList(args.converted_sample_root+'CVQ_1024')
        CVQ1024_wav_fpaths = [path for path in tqdm(CVQ1024_wav_fpaths) if spk in path.split('_')[2]]
        
        CVQ2048_wav_fpaths = fList(args.converted_sample_root+'CVQ_2048')
        CVQ2048_wav_fpaths = [path for path in tqdm(CVQ2048_wav_fpaths) if spk in path.split('_')[2]]
        
        VQ2048_wav_fpaths = fList(args.converted_sample_root+'VQVC_2048')
        VQ2048_wav_fpaths = [path for path in tqdm(VQ2048_wav_fpaths) if spk in path.split('_')[2]]

        

        # Load the audio file
        paths_list = [real_wav_fpaths, VQ256_wav_fpaths, CVQ256_wav_fpaths, VQ512_wav_fpaths, CVQ512_wav_fpaths, VQ2048_wav_fpaths, CVQ2048_wav_fpaths]
        total_score = []
        for paths in tqdm(paths_list):
            scores = get_wer(model, processor, gt_txt_fpaths, paths)
            total_score.append(scores)

total_scores = np.array(total_scores)
total_avg = np.mean(total_scores, axis=0)
total_max = np.max(total_scores, axis=0)
total_min = np.min(total_scores, axis=0)

indices = ['Real','VQVC_256', 'CVQ_256', 'VQVC_512', 'CVQ_512', 'VQVC_1024', 'CVQ_1024', 'VQVC_2024', 'CVQ_2024']
# scores = [real_score, VQ256_score, CVQ256_score, VQ512_score, CVQ512_score, VQ1024_score, CVQ1024_score, VQ2048_score, CVQ2048_score]
colors = ["skyblue", "lightgreen", "lightcoral", "lightsalmon", "lightblue", "lightpink", "lightcoral", "lightgreen","lightsalmon"]

fig, _ = plt.subplots(figsize=(6, 6))
plt.bar(indices, total_avg, color=colors)
plt.errorbar(indices, total_avg, yerr=(total_avg-total_min, total_max-total_avg), fmt='none', ecolor='red', capsize=5, label="Error Bars")
plt.xlabel("Models")
plt.ylabel("Similarity scores")
plt.xticks(rotation=30)
plt.title("Similarity")
plt.savefig(args.root_dir+'fake_detection_errorBar_1.png')
print('END')