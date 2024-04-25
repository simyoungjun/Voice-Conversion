import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.io import wavfile
from tqdm import tqdm


def process(wav_name):
    # speaker 's5', 'p280', 'p315' are excluded,
    speaker = wav_name[:4]
    wav_path = os.path.join(args.in_dir, wav_name)
    # if os.path.exists(wav_path) and '_mic2.flac' in wav_path:
    if os.path.exists(wav_path):
        os.makedirs(os.path.join(args.out_dir1, speaker), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)
        wav, sr = librosa.load(wav_path)
        wav, _ = librosa.effects.trim(wav, top_db=20)
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        wav1 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr1)
        wav2 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr2)
        save_name = wav_name
        save_path1 = os.path.join(args.out_dir1, save_name)
        save_path2 = os.path.join(args.out_dir2, save_name)
        wavfile.write(
            save_path1,
            args.sr1,
            (wav1 * np.iinfo(np.int16).max).astype(np.int16)
        )
        wavfile.write(
            save_path2,
            args.sr2,
            (wav2 * np.iinfo(np.int16).max).astype(np.int16)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr1", type=int, default=16000, help="sampling rate")
    parser.add_argument("--sr2", type=int, default=22050, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="/home/sim/VoiceConversion/FreeVC/data_sample", help="path to source dir")
    parser.add_argument("--out_dir1", type=str, default="/home/sim/VoiceConversion/FreeVC/data_sample/sample-16k", help="path to target dir")
    parser.add_argument("--out_dir2", type=str, default="/home/sim/VoiceConversion/FreeVC/data_sample/samplee-22k", help="path to target dir")
    args = parser.parse_args()

    pool = Pool(processes=cpu_count()-2)


    for _ in tqdm(pool.imap_unordered(process, os.listdir(args.in_dir))):
        pass
