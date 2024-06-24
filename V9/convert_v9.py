import sys, os

sys.path.append('/home/sim/VoiceConversion/FreeVC')
# sys.path.append('/home/sim/VoiceConversion/torch_hpss')

import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import json

# import torch_hpss
import numpy as np

import utils

from models_v9_concat import SynthesizerTrn
# from models_v9 import SynthesizerTrn

from mel_processing import mel_spectrogram_torch, spectrogram_torch, spec_to_mel_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import shutil


import logging
logging.getLogger('numba').setLevel(logging.WARNING)

def get_path(*args):
        return os.path.join('', *args)
    
if __name__ == "__main__":
    
    model_name = 'V9_VQ1024_res_slice_concat'
    # meta_data = 'LibriTTS_unseen'
    meta_data = 'VCTK_seen'
    
    
    # out_name = 'VCTK_seen_60(1000)'
    
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--ptfile", type=str, default="/shared/NAS_SSD/sim/Checkpoints/logs/V8_VQ2048_SrcRef/G_100000.pth", help="path to pth file")
    parser.add_argument("--ptfile", type=str, default="/shared/racoon_fast/sim/Checkpoints/logs/V9_VQ1024_res_slice_concat/G_44000.pth")
    
    parser.add_argument("--hpfile", type=str, default=f"/shared/racoon_fast/sim/Checkpoints/logs/{model_name}/{model_name}.json", help="path to json config file")
    # parser.add_argument("--hpfile", type=str, default=f"/shared/NAS_SSD/sim/Checkpoints/logs/{model_name}/{model_name}.json", help="path to json config file")
    
    # parser.add_argument("--txtpath", type=str, default="/home/sim/VoiceConversion/conversion_metas/VCTK_seen_pairs(1000).txt", help="path to txt file")
    parser.add_argument("--txtpath", type=str, default=f"/home/sim/VoiceConversion/conversion_metas/{meta_data}_pairs(1000).txt", help="path to txt file")
    
    # parser.add_argument("--outdir", type=str, default=f"/shared/racoon_fast/sim/results/{model_name}/output/VCTK_seen_94(1000)", help="path to output dir")
    parser.add_argument("--outdir", type=str, default=f"/shared/racoon_fast/sim/results/{model_name}/output/{meta_data}_44(1000)_no_trim", help="path to output dir")
    
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

            titles.append('src;'+src.split('/')[-1][:-4]+'&tgt;'+tgt.split('/')[-1][:-4])
            
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            src = src.replace('vctk-16k', 'vctk-16k_no_trim')
            tgt = tgt.replace('vctk-16k', 'vctk-16k_no_trim')
 
            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            src_c = utils.get_content(cmodel, wav_src, layer=6)
            
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
            tgt_c = utils.get_content(cmodel, wav_tgt, layer=6)
            

            audio = net_g.convert(src_c, tgt_c)
            audio = audio[0][0].data.cpu().float().numpy()
            
            save_dir = os.path.join(args.outdir, f"{title}")
            os.makedirs(save_dir, exist_ok=True)
            
            write(os.path.join(save_dir, f"C!{title}.wav"), hps.data.sampling_rate, audio)
            
            shutil.copy2(src, f"{save_dir}/S!{src.split('/')[-1]}")
            shutil.copy2(tgt, f"{save_dir}/T!{tgt.split('/')[-1]}")
        
