import sys, os

sys.path.append('/home/sim/VoiceConversion/FreeVC')
sys.path.append('/home/sim/VoiceConversion/torch_hpss')

import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import json
import numpy as np
import librosa 

import torch_hpss
from scipy.io.wavfile import read

import utils
from models_v3 import SynthesizerTrn
from mel_processing import mel_spectrogram_torch, spectrogram_torch, spec_to_mel_torch
# from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import shutil


import logging
logging.getLogger('numba').setLevel(logging.WARNING)

def get_path(*args):
        return os.path.join('', *args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="/home/sim/VoiceConversion/V3/config.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="/home/sim/VoiceConversion/V3/G_100000.pth", help="path to pth file")
    parser.add_argument("--txtpath", type=str, default="/home/sim/VoiceConversion/conversion_metas/unseen_pairs.txt", help="path to txt file")
    parser.add_argument("--outdir", type=str, default="/home/sim/VoiceConversion/V3/output/VCTK_100", help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    # print("Loading WavLM for content...")
    # cmodel = utils.get_cmodel(0)
    
    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    print("Processing text...")
        
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as file:
        for rawline in file.readlines()[:]:
            title, tgt, src = rawline.strip().split("|")
            # titles.append(f'{title.split('/')[-1][:-4]}_from_{src.split('/')[-2]}')
            titles.append('src:'+src.split('/')[-1][:-4]+'&tgt:'+tgt.split('/')[-1][:-4])

            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            # tgt

            sampling_rate, wav_tgt = read(tgt)
            
            # wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            
            if hps.model.use_spk:
                # print("use spk?")
                # g_tgt = smodel.embed_utterance(wav_tgt)
                # g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
                pass
            else:
                wav_tgt_norm = wav_tgt.astype(np.float32)/hps.data.max_wav_value
                wav_tgt_norm = torch.from_numpy(wav_tgt_norm).unsqueeze(0).cuda()
                
                spec_tgt = spectrogram_torch(wav_tgt_norm, hps.data.filter_length,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    center=False)
                
                hpss_module = torch_hpss.HPSS(kernel_size=31, reduce_method='median').cuda()
                res_tgt = hpss_module(spec_tgt.unsqueeze(1))
                spec_H = res_tgt['harm_spec'].squeeze(1)
                
                mel_H = spec_to_mel_torch(
                    spec_H, 
                    hps.data.filter_length, 
                    hps.data.n_mel_channels, 
                    hps.data.sampling_rate,
                    hps.data.mel_fmin, 
                    hps.data.mel_fmax) 

            # src
            sampling_rate, wav_src = read(src)
            
            wav_src_norm = wav_src.astype(np.float32)/hps.data.max_wav_value
            wav_src_norm = torch.from_numpy(wav_src_norm).unsqueeze(0).cuda()
            spec_src = spectrogram_torch(wav_src_norm, hps.data.filter_length,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    center=False)               
            
            res_tgt = hpss_module(spec_src.unsqueeze(1))
            spec_P = res_tgt['perc_spec'].squeeze(1)
            

            if hps.model.use_spk:
                # audio = net_g.infer(c, g=g_tgt)
                pass
            else:
                audio = net_g.infer(c=spec_P, mel=mel_H)
            audio = audio[0][0].data.cpu().float().numpy()
            
            save_dir = os.path.join(args.outdir, f"{title}")
            os.makedirs(save_dir, exist_ok=True)
            
            write(os.path.join(save_dir, f"C|{title}.wav"), hps.data.sampling_rate, audio)
            
            shutil.copy2(src, f"{save_dir}/S|{src.split('/')[-1]}")
            shutil.copy2(tgt, f"{save_dir}/T|{tgt.split('/')[-1]}")
        
