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

import torch_hpss
import numpy as np

import utils

from models_v6_H import SynthesizerTrn
from mel_processing import mel_spectrogram_torch, spectrogram_torch, spec_to_mel_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import shutil


import logging
logging.getLogger('numba').setLevel(logging.WARNING)

def get_path(*args):
        return os.path.join('', *args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="/home/sim/VoiceConversion/V6/freevc_v6_H.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="/shared/racoon_fast/sim/Checkpoints/logs/V6_H/G_167000.pth", help="path to pth file")
    parser.add_argument("--txtpath", type=str, default="/home/sim/VoiceConversion/conversion_metas/LibriTTS_unseen_pairs.txt", help="path to txt file")
    
    parser.add_argument("--outdir", type=str, default="/home/sim/VoiceConversion/V6/output/Libri-H_167", help="path to output dir")
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

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)
    
    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    print("Processing text...")
    
        
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as file:
        for rawline in file.readlines()[:]:
            title, tgt, src = rawline.strip().split("|")

            titles.append('src:'+src.split('/')[-1][:-4]+'&tgt:'+tgt.split('/')[-1][:-4])
            
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            # tgt
            # wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            # wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            wav_tgt, sampling_rate = utils.load_wav_to_torch(tgt)
            if hps.model.use_spk:
                # print("use spk?")
                # g_tgt = smodel.embed_utterance(wav_tgt)
                # g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
                pass
            else:
                
                # wav_tgt_norm = wav_tgt.astype(np.float32)/hps.data.max_wav_value
                # wav_tgt_norm = torch.from_numpy(wav_tgt_norm).unsqueeze(0).cuda()
                wav_tgt_norm = wav_tgt/hps.data.max_wav_value
                wav_tgt_norm = wav_tgt_norm.unsqueeze(0).cuda()                
                spec_tgt = spectrogram_torch(wav_tgt_norm, hps.data.filter_length,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    center=False)
                
                hpss_module = torch_hpss.HPSS(kernel_size=31, reduce_method='median').cuda()
                res_tgt = hpss_module(spec_tgt.unsqueeze(1))
                spec_H = res_tgt['harm_spec'].squeeze(1)
                
                mel = spec_to_mel_torch(
                    spec_tgt, 
                    hps.data.filter_length, 
                    hps.data.n_mel_channels, 
                    hps.data.sampling_rate,
                    hps.data.mel_fmin, 
                    hps.data.mel_fmax) 
                
                mel_H_tgt = spec_to_mel_torch(
                    spec_H, 
                    hps.data.filter_length, 
                    hps.data.n_mel_channels, 
                    hps.data.sampling_rate,
                    hps.data.mel_fmin, 
                    hps.data.mel_fmax) 

            # src
            wav_src, sampling_rate = utils.load_wav_to_torch(src)
            wav_src_norm = wav_src/hps.data.max_wav_value
            wav_src_norm = wav_src_norm.unsqueeze(0).cuda()                
            spec_src = spectrogram_torch(wav_src_norm, hps.data.filter_length,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    center=False)
            spec_src
            
            # fig = utils.draw_converted_melspectrogram(mel_tgt, c, mel_converted, mel_src_code, mel_src_style, mel_ref_code, mel_ref_style)
            # fig.savefig(get_path(args.converted_sample_dir, "contents_{}_style_{}.png".format(src_wav_name.replace(".wav", ""), ref_wav_name.replace(".wav", ""))))	
            
            
            if hps.model.use_spk:
                # audio = net_g.infer(c, g=g_tgt)
                pass
            else:
                audio = net_g.infer(spec_src, mel=mel_H_tgt)
            audio = audio[0][0].data.cpu().float().numpy()
            
            save_dir = os.path.join(args.outdir, f"{title}")
            os.makedirs(save_dir, exist_ok=True)
            
            write(os.path.join(save_dir, f"C|{title}.wav"), hps.data.sampling_rate, audio)
            
            shutil.copy2(src, f"{save_dir}/S|{src.split('/')[-1]}")
            shutil.copy2(tgt, f"{save_dir}/T|{tgt.split('/')[-1]}")
        
