import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.io import wavfile
from tqdm import tqdm
import random

def process(chapter, speaker):
    # speaker 's5', 'p280', 'p315' are excluded,
    for file in os.listdir(chapter):
        wav_path = os.path.join(args.in_dir, speaker, chapter, file)
        if '.wav' in file:
            os.makedirs(os.path.join(args.out_dir1, speaker), exist_ok=True)
            os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)
            wav, sr = librosa.load(wav_path)
            wav, _ = librosa.effects.trim(wav, top_db=20)
            peak = np.abs(wav).max()
            if peak > 1.0:
                wav = 0.98 * wav / peak
            wav1 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr1)
            wav2 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr2)
            # save_name = wav_name.replace("_mic2.flac", ".wav")
            save_path1 = os.path.join(args.out_dir1, speaker, file)
            save_path2 = os.path.join(args.out_dir2, speaker, file)
            
            if not os.path.exists(save_path1):
                wavfile.write(
                    save_path1,
                    args.sr1,
                    (wav1 * np.iinfo(np.int16).max).astype(np.int16)
                )
            else:
                pass
            wavfile.write(
                save_path2,
                args.sr2,
                (wav2 * np.iinfo(np.int16).max).astype(np.int16)
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr1", type=int, default=16000, help="sampling rate")
    parser.add_argument("--sr2", type=int, default=24000, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="/shared/racoon_fast/youngjun/LibriTTS/train-clean-100", help="path to source dir")
    parser.add_argument("--out_dir1", type=str, default="/shared/racoon_fast/youngjun/LibriTTS/preprocessed/LibriTTS-16k", help="path to target dir")
    parser.add_argument("--out_dir2", type=str, default="/shared/racoon_fast/youngjun/LibriTTS/preprocessed/LibriTTS-22k", help="path to target dir")
    args = parser.parse_args()

    # pool = Pool(processes=cpu_count()-2)
    # pool = Pool(processes=1)
    

    for speaker in tqdm(os.listdir(args.in_dir)):
        spk_dir = os.path.join(args.in_dir, speaker)
        for chapter in os.listdir(spk_dir):
            process(os.path.join(args.in_dir, speaker, chapter), speaker)