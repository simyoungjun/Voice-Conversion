import sys, os

sys.path.append('/home/sim/VoiceConversion/FreeVC')

import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import json
import numpy as np
import shutil

import utils
from models_v2 import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
# from wavlm import WavLM, WavLMConfig
# from speaker_encoder.voice_encoder import SpeakerEncoder
# import logging
# logging.getLogger('numba').setLevel(logging.WARNING)
from scipy.io.wavfile import read

def get_path(*args):
        return os.path.join('', *args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="/home/sim/VoiceConversion/V2/logs/config.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="/home/sim/VoiceConversion/V2/logs/G_500000.pth", help="path to pth file")
    # parser.add_argument("--txtpath", type=str, default="/home/sim/VoiceConversion/conversion_metas/LibriTTS_pairs.txt", help="path to txt file")
    parser.add_argument("--txtpath", type=str, default="/home/sim/VoiceConversion/conversion_metas/unseen_pairs.txt", help="path to txt file")
    # parser.add_argument("--outdir", type=str, default="/home/sim/VoiceConversion/V2/output/LibriTTS_100", help="path to output dir")
    parser.add_argument("--outdir", type=str, default="/home/sim/VoiceConversion/V2/output/VCTK_500", help="path to output dir")
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
    
    # unseen_speaker_list = ["p226", "p256", "p266", "p297", "p323", "p376"]
    # file_num = 100
    # conversion_path_dir = '/root/sim/VoiceConversion/FreeVC/conversion_metas/unseen/'
    # titles = []
    # for spk in unseen_speaker_list:
    #     with open(conversion_path_dir+spk+'.json', 'r') as json_file:
    #         conversion_paths = json.load(json_file)

    #     tgts = conversion_paths['ref_paths'][:file_num]
    #     srcs = conversion_paths['src_paths'][:file_num]
    #     # titles.append(spk+'1')
        
    #     titles = [get_path(args.outdir, "{}_{}_from_{}.wav".format(tgt.split("/")[-1].split('_')[0], src.split("/")[-1].split('_')[1].split('.')[0], src.split("/")[-1].split('_')[0])) for src, tgt in zip(srcs, tgts)]
        
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
            # wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            sampling_rate, wav_tgt = read(tgt)
            
            # wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            if hps.model.use_spk:
                print("use spk?")
                g_tgt = smodel.embed_utterance(wav_tgt)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
            else:
                # tgt_audio_norm = wav_tgt.astype(np.float32)
                tgt_audio_norm = wav_tgt.astype(np.float32)/hps.data.max_wav_value
                
                tgt_audio_norm_H, tgt_audio_norm_P = librosa.effects.hpss(tgt_audio_norm)
                tgt_audio_norm_H = torch.from_numpy(tgt_audio_norm_H).unsqueeze(0)
                
                tgt_mel_H = mel_spectrogram_torch(
                    tgt_audio_norm_H, 
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )
                tgt_mel_H = tgt_mel_H.cuda(0)
                           
                # wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
                # mel_tgt = mel_spectrogram_torch(
                #     wav_tgt, 
                #     hps.data.filter_length,
                #     hps.data.n_mel_channels,
                #     hps.data.sampling_rate,
                #     hps.data.hop_length,
                #     hps.data.win_length,
                #     hps.data.mel_fmin,
                #     hps.data.mel_fmax
                # )
            # src
            # wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            sampling_rate, wav_src = read(src)
            src_audio_norm = wav_src.astype(np.float32)/hps.data.max_wav_value
            src_audio_norm_H, src_audio_norm_P = librosa.effects.hpss(src_audio_norm)
            src_audio_norm_P = torch.from_numpy(src_audio_norm_P).unsqueeze(0)
            
            src_mel_P = mel_spectrogram_torch(
            src_audio_norm_P, 
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
            )
            src_mel_P = src_mel_P.cuda(0)
            
            # wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            # c = utils.get_content(cmodel, wav_src)
            
            # fig = utils.draw_converted_melspectrogram(mel_tgt, c, mel_converted, mel_src_code, mel_src_style, mel_ref_code, mel_ref_style)
            # fig.savefig(get_path(args.converted_sample_dir, "contents_{}_style_{}.png".format(src_wav_name.replace(".wav", ""), ref_wav_name.replace(".wav", ""))))	
            
            
            if hps.model.use_spk:
                # audio = net_g.infer(c, g=g_tgt)
                pass
            else:
                audio = net_g.infer(c=src_mel_P, mel=tgt_mel_H)
            audio = audio[0][0].data.cpu().float().numpy()
            
            save_dir = os.path.join(args.outdir, f"{title}")
            os.makedirs(save_dir, exist_ok=True)
            
            write(os.path.join(save_dir, f"C|{title}.wav"), hps.data.sampling_rate, audio)
            
            shutil.copy2(src, f"{save_dir}/S|{src.split('/')[-1]}")
            shutil.copy2(tgt, f"{save_dir}/T|{tgt.split('/')[-1]}")
        
