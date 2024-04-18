from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

# DEMO 05: In this demo we'll show how we can achieve a modest form of fake speech detection with 
# Resemblyzer. This method assumes you have some reference audio for the target speaker that you 
# know is real, so it is not a universal fake speech detector on its own.
# In the audio data directory we have 18 segments of Donald Trump. 12 are real and extracted from
# actual speeches, while the remaining 6 others are fake and generated by various users on 
# youtube, with a high discrepancy of voice cloning quality and naturalness achieved. We will 
# take 6 segments of real speech as ground truth reference and compare those against the 12 
# remaining. Those segments are selected at random, so will run into different results every time
# you run the script, but they should be more or less consistent.
# Using the voice of Donald Trump is merely a matter of convenience, as several fake speeches 
# with his voice were already put up on youtube. This choice was not politically motivated.

# # VCTK
models_paths = [
    "/home/sim/VoiceConversion/V6/output/VCTK-p_557",
    "/home/sim/VoiceConversion/V6/output/VCTK_250",
    # "/home/sim/VoiceConversion/V5_5/output/VCTK_216",
    # "/home/sim/VoiceConversion/V5_3/output/VCTK_89",
    # "/home/sim/VoiceConversion/V5_2/output/VCTK_257",
    # ,"/home/sim/VoiceConversion/V5/output/VCTK_272"
    # ,"/home/sim/VoiceConversion/V5/output/VCTK_196"
    # ,"/home/sim/VoiceConversion/V5/output/VCTK_49"
    "/home/sim/VoiceConversion/FreeVC/output/freevc/VCTK_s-0",
    #             "/home/sim/VoiceConversion/V4/output/VCTK_500",
    #             "/home/sim/VoiceConversion/V3/output/VCTK_100",
    #             "/home/sim/VoiceConversion/YourTTS/output"
                ]

# #LibriTTS
# models_paths = ["/home/sim/VoiceConversion/FreeVC/output/freevc/LibriTTS_s-0"
#     ,"/home/sim/VoiceConversion/V5/output/LibriTTS_186"
#     # ,"/home/sim/VoiceConversion/FreeVC/output/freevc/VCTK_s-0",
#     #             "/home/sim/VoiceConversion/V4/output/VCTK_500",
#     #             "/home/sim/VoiceConversion/V3/output/VCTK_100",
#     #             "/home/sim/VoiceConversion/YourTTS/output"
#                 ]

model_wav_list = []
names = []
for model_path in models_paths:
	names.append(model_path.split('/')[4])
	tgt_list = []
	cvt_list = []
	for root, dirs, files in os.walk(model_path):
		for file in files:
			if "T" in file:
				tgt_list.append(os.path.join(root, file))
				
			elif "C" in file:
				cvt_list.append(os.path.join(root, file))
	
	model_wav_list.append([tgt_list, cvt_list])
 
scores = []
for wav_fpaths in model_wav_list:	 
	## Load and preprocess the audio

	tgt_wavs = [preprocess_wav(wav_fpath) for wav_fpath in \
			tqdm(wav_fpaths[0], "Preprocessing wavs", len(wav_fpaths[0]), unit=" utterances")]
	cvt_wavs = [preprocess_wav(wav_fpath) for wav_fpath in \
			tqdm(wav_fpaths[1], "Preprocessing wavs", len(wav_fpaths[1]), unit=" utterances")]


	## Compute the embeddings
	encoder = VoiceEncoder()
	embeds = np.array([encoder.embed_utterance(wav) for wav in cvt_wavs])
	gt_embeds = np.array([encoder.embed_utterance(wav) for wav in tgt_wavs])
 
	# speakers = np.array([fpath.parent.name for fpath in wav_fpaths])
	# names = np.array([fpath.stem for fpath in wav_fpaths])


	# # Take 6 real embeddings at random, and leave the 6 others for testing
	# gt_indices = np.random.choice(*np.where(speakers == "real"), 6, replace=False) 
	# mask = np.zeros(len(embeds), dtype=bool)
	# mask[gt_indices] = True
	# gt_embeds = embeds[mask]
	# gt_names = names[mask]
	# gt_speakers = speakers[mask]
	# embeds, speakers, names = embeds[~mask], speakers[~mask], names[~mask]


	## Compare all embeddings against the ground truth embeddings, and compute the average similarities.
	score = (gt_embeds @ embeds.T)
	mask = np.eye(len(wav_fpaths[0]), dtype=int)
	score = score*mask
	score_avg = np.sum(score)/len(wav_fpaths[0])
	scores.append(score_avg)
	
	# score = (gt_embeds @ gt_embeds.T)
	# mask = np.eye(len(wav_fpaths[0]), dtype=int)
	# score = score*mask
	# score_avg = np.mean(score)
	# scores.append(score_avg)
 
# Order the scores by decreasing order
sort = np.argsort(scores)[::-1]
scores, names = np.array(scores), np.array(names)
scores, names = scores[sort], names[sort]
print(scores, names)

## Plot the scores
fig, _ = plt.subplots(figsize=(6, 6))
indices = np.arange(len(scores))
plt.axhline(0.84, ls="dashed", label="Prediction threshold", c="black")
plt.bar(indices, scores)
plt.legend()
plt.xticks(indices, names, rotation="vertical", fontsize=8)
plt.xlabel("Youtube video IDs")
plt.ylim(0, 1)
plt.ylabel("Similarity to ground truth")
fig.subplots_adjust(bottom=0.25)
# plt.show()
plt.savefig('/home/sim/VoiceConversion/Evaluation/EER.png')