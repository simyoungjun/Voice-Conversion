import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import json
import shutil

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
import numpy as np
# logging.getLogger('numba').setLevel(logging.WARNING)



def preprocess(wav, trim=False):
    if trim == True:
        wav, _ = librosa.effects.trim(wav, top_db=20)
    
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = 0.98 * wav / peak
    wav1 = librosa.resample(wav, orig_sr=48000, target_sr=16000)
    return wav1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="/home/sim/VoiceConversion/FreeVC/logs/freevc/config.json", help="path to json config file")
    # parser.add_argument("--ptfile", type=str, default="/home/sim/VoiceConversion/FreeVC/logs/freevc-s/G_0.pth", help="path to pth file")
    parser.add_argument("--ptfile", type=str, default="/home/sim/VoiceConversion/FreeVC/logs/freevc/freevc.pth", help="path to pth file")
    parser.add_argument("--txtpath", type=str, default="/home/sim/VoiceConversion/conversion_metas/VCTK_seen_pairs(1000).txt", help="path to txt file")
    # parser.add_argument("--txtpath", type=str, default="/home/sim/VoiceConversion/conversion_metas/VCTK_seen_pairs(1000).txt", help="path to txt file")
    # parser.add_argument("--outdir", type=str, default="/home/sim/VoiceConversion/FreeVC/output/freevc/VCTK_seen(1000)", help="path to output dir")
    parser.add_argument("--outdir", type=str, default="/shared/racoon_fast/sim/results/FreeVC/output/freevc/VCTK_seen_no_trim_wav48_2(1000)", help="path to output dir")
    
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    
    # GPU_NUM = 0 # 원하는 GPU 번호 입력
    # device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(device) # change allocation of current GPU
    
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
        smodel = SpeakerEncoder('/home/sim/VoiceConversion/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    print("Processing text...")
    # titles, srcs, tgts = [], [], []
    # with open(args.txtpath, "r") as f:
    #     for rawline in f.readlines():
    #         title, src, tgt = rawline.strip().split("|")
    #         titles.append(title)
    #         srcs.append(src)
    #         tgts.append(tgt)
    
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as file:
        for rawline in file.readlines()[:]:
            title, tgt, src = rawline.strip().split("|")
            # titles.append(f'{title.split('/')[-1][:-4]}_from_{src.split('/')[-2]}')
            # titles.append(title.split('/')[-1][:-4]+'_from_'+src.split('/')[-2])
            # if "LibriTTS" in tgt:
            titles.append('src;'+src.split('/')[-1][:-4]+'&tgt;'+tgt.split('/')[-1][:-4])
            
            srcs.append(src)
            tgts.append(tgt)
            # elif "VCTK" in tgt:
            #     titles.append('src;'+src.split('/')[-1][:-4]+'&tgt;'+tgt.split('/')[-1][:-4])
                
            #     srcs.append(src)
            #     tgts.append(tgt)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            tgt = tgt.replace('preprocessed/vctk-16k/', 'wav48_silence_trimmed/').replace('.wav', '_mic1.flac')
            src = src.replace('preprocessed/vctk-16k/', 'wav48_silence_trimmed/').replace('.wav', '_mic1.flac')
            # tgt
            # wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.load(tgt, sr=48000)
            wav_tgt = preprocess(wav_tgt, trim=True)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            if hps.model.use_spk:
                g_tgt = smodel.embed_utterance(wav_tgt)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
      
            else:
                wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
                mel_tgt = mel_spectrogram_torch(
                    wav_tgt, 
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )
            # src
            wav_src, _ = librosa.load(src, sr=48000)
            wav_src = preprocess(wav_src)
            wav_src_np = wav_src.copy()
            
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            c = utils.get_content(cmodel, wav_src)
            
            # fig = utils.draw_converted_melspectrogram(mel_tgt, c, mel_converted, mel_src_code, mel_src_style, mel_ref_code, mel_ref_style)
            # fig.savefig(get_path(args.converted_sample_dir, "contents_{}_style_{}.png".format(src_wav_name.replace(".wav", ""), ref_wav_name.replace(".wav", ""))))	
            
            
            if hps.model.use_spk:
                audio = net_g.infer(c, g=g_tgt)
            else:
                audio = net_g.infer(c, mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()
            
            save_dir = os.path.join(args.outdir, f"{title}")
            os.makedirs(save_dir, exist_ok=True)
            
            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(os.path.join(args.outdir, "{}.wav".format(timestamp+"_"+title)), hps.data.sampling_rate, audio)
            else:
                write(os.path.join(save_dir, f"C!{title}.wav"), hps.data.sampling_rate, audio)
            
            tgt_write = (wav_tgt * np.iinfo(np.int16).max).astype(np.int16)
            src_write = (wav_src_np * np.iinfo(np.int16).max).astype(np.int16)
            
            write(f"{save_dir}/S!{src.split('/')[-1].replace('_mic1.flac', '.wav')}", hps.data.sampling_rate, src_write)
            write(f"{save_dir}/T!{tgt.split('/')[-1].replace('_mic1.flac', '.wav')}", hps.data.sampling_rate, tgt_write)
            
            # shutil.copy2(src, f"{save_dir}/shut_S!{src.split('/')[-1]}")
            # shutil.copy2(tgt, f"{save_dir}/shut_T!{tgt.split('/')[-1]}")
            # if "LibriTTS" in tgt:
            #     shutil.copy2(src.replace('preprocessed/LibriTTS-16k', 'train-clean-100').replace('wav', 'original.txt'), f"{save_dir}/text.txt")
            
