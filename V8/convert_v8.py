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

from models_v8 import SynthesizerTrn
from mel_processing import mel_spectrogram_torch, spectrogram_torch, spec_to_mel_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import shutil


import logging
logging.getLogger('numba').setLevel(logging.WARNING)

def get_path(*args):
        return os.path.join('', *args)
    
if __name__ == "__main__":
    
    model_name = 'V8_VQ1024_code_loss'
    # meta_data = 'LibriTTS_unseen'
    meta_data = 'VCTK_seen'
    
    # out_name = 'VCTK_seen_60(1000)'
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--hpfile", type=str, default=f"/shared/racoon_fast/sim/Checkpoints/logs/V8_VQ256_no_affine_cond/{model_name}.json", help="path to json config file")
    parser.add_argument("--hpfile", type=str, default=f"/shared/racoon_fast/sim/Checkpoints/logs/{model_name}/{model_name}.json", help="path to json config file")
    
    parser.add_argument("--ptfile", type=str, default="/shared/racoon_fast/sim/Checkpoints/logs/V8_VQ1024_code_loss/G_100000.pth", help="path to pth file")
    # parser.add_argument("--ptfile", type=str, default="/shared/racoon_fast/sim/Checkpoints/logs/V8_VQ256_no_affine_cond/G_60000.pth", help="path to pth file")
    
    # parser.add_argument("--txtpath", type=str, default="/home/sim/VoiceConversion/conversion_metas/VCTK_seen_pairs(1000).txt", help="path to txt file")
    parser.add_argument("--txtpath", type=str, default=f"/home/sim/VoiceConversion/conversion_metas/{meta_data}_pairs(1000).txt", help="path to txt file")
    
    # parser.add_argument("--outdir", type=str, default=f"/shared/racoon_fast/sim/results/{model_name}/output/VCTK_seen_94(1000)", help="path to output dir")
    parser.add_argument("--outdir", type=str, default=f"/shared/racoon_fast/sim/results/{model_name}/output/{meta_data}_100(1000)", help="path to output dir")
    
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
            # # tgt
            # # wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            # # wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            # wav_tgt, sampling_rate = utils.load_wav_to_torch(tgt)
            # if hps.model.use_spk:
            #     # print("use spk?")
            #     # g_tgt = smodel.embed_utterance(wav_tgt)
            #     # g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
            #     pass
            # else:
                
            #     # wav_tgt_norm = wav_tgt.astype(np.float32)/hps.data.max_wav_value
            #     # wav_tgt_norm = torch.from_numpy(wav_tgt_norm).unsqueeze(0).cuda()
            #     wav_tgt_norm = wav_tgt/hps.data.max_wav_value
            #     wav_tgt_norm = wav_tgt_norm.unsqueeze(0).cuda()                
            #     spec_tgt = spectrogram_torch(wav_tgt_norm, hps.data.filter_length,
            #         hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
            #         center=False)
                
            #     # hpss_module = torch_hpss.HPSS(kernel_size=31, reduce_method='median').cuda()
            #     # res_tgt = hpss_module(spec_tgt.unsqueeze(1))
            #     # spec_H = res_tgt['harm_spec'].squeeze(1)
                
            #     mel = spec_to_mel_torch(
            #         spec_tgt, 
            #         hps.data.filter_length, 
            #         hps.data.n_mel_channels, 
            #         hps.data.sampling_rate,
            #         hps.data.mel_fmin, 
            #         hps.data.mel_fmax) 
 
            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            src_c = utils.get_content(cmodel, wav_src, layer=6)
            
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
            tgt_c = utils.get_content(cmodel, wav_tgt, layer=6)
            
            # src_c_filename = src.replace(".wav", ".pt")
            # src_c_filename = src_c_filename.replace("vctk-16k", "wavlm-6L")
            # src_c = torch.load(src_c_filename).cuda()
            
            # tgt_c_filename = tgt.replace(".wav", ".pt")
            # tgt_c_filename = tgt_c_filename.replace("vctk-16k", "wavlm-6L")
            # tgt_c = torch.load(tgt_c_filename).cuda()
            
            # fig = utils.draw_converted_melspectrogram(mel_tgt, c, mel_converted, mel_src_code, mel_src_style, mel_ref_code, mel_ref_style)
            # fig.savefig(get_path(args.converted_sample_dir, "contents_{}_style_{}.png".format(src_wav_name.replace(".wav", ""), ref_wav_name.replace(".wav", ""))))	
            
            

            audio = net_g.convert(src_c, tgt_c)
            audio = audio[0][0].data.cpu().float().numpy()
            
            save_dir = os.path.join(args.outdir, f"{title}")
            os.makedirs(save_dir, exist_ok=True)
            
            write(os.path.join(save_dir, f"C!{title}.wav"), hps.data.sampling_rate, audio)
            
            shutil.copy2(src, f"{save_dir}/S!{src.split('/')[-1]}")
            shutil.copy2(tgt, f"{save_dir}/T!{tgt.split('/')[-1]}")
        
