import sys, os

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

sys.path.append('/home/sim/VoiceConversion/torch_hpss')

import torch_hpss
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
# import GPUtil

import librosa

sys.path.append('/home/sim/VoiceConversion/FreeVC')
import commons
import utils
# 이부분 바뀜
from data_utils_v6_3 import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from models_v6_3 import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
  vq_loss,
  info_nce_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

import wandb


torch.backends.cudnn.benchmark = True
global_step = 0
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# os.environ["TORCH_USE_CUDA_DSA"] = '1'

def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."
  # torch.multiprocessing.set_start_method('spawn')
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="/home/sim/VoiceConversion/V6_3/freevc_v6_3.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, default="V6_3_noIN",
                      help='Model name')
  args = parser.parse_args()
  
  hps = utils.get_hparams(args=args)

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = hps.train.port

  # run(rank=0, ...)
  # run(0, n_gpus, hps)
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  
  global global_step
  if rank == 0:
    if hps.setting.log_wandb:
      wandb.init('p')
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    # writer = SummaryWriter(log_dir=hps.model_dir)
    # writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    writer = None
    writer_eval = None
  
  torch.cuda.set_device(rank)

  
  dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23459', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)

  

  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate(hps)
  
  num_workers=hps.train.num_workers
  
  train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
    eval_loader = DataLoader(eval_dataset, num_workers=num_workers, shuffle=True,
        batch_size=hps.train.batch_size, pin_memory=False,
        drop_last=False, collate_fn=collate_fn)

  net_g = SynthesizerTrn(
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  
  net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)#)
  net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
  # net_g = DDP(net_g, device_ids=[rank])#)
  # net_d = DDP(net_d, device_ids=[rank])

  
  try:
    net_g, optim_g, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    net_d, optim_d, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  '''
  Args:
    c: content embedding (ssl output) (N, z_dim, T) : (64, 1024, T)
    g:  speaker encoder output (speaker embedding)  (N, z_dim) : (64, 256)
    spec: spectrogram (N, F, T) default: (64, 641, T), F: Frequency bin
    mel: mel (N, F, T) default: (64, 80, T)
  
  '''
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  
  hpss_module = torch_hpss.HPSS(kernel_size=31, reduce_method='median').cuda(rank)
  
  for batch_idx, items in enumerate(train_loader):
    # GPUtil.showUtilization()
    if hps.model.use_spk:
      # 안씀
      c, spec, y, spk = items
      g = spk.cuda(rank)
    else:
    # V5
      c, spec, y = items
      g = None
    
    ###
    spec, y = spec.cuda(rank), y.cuda(rank)
    # c = c.cuda(rank)
    # V5_3
    c = spec
    
    res = hpss_module(spec.unsqueeze(1))
    spec_H, spec_P = res['harm_spec'].squeeze(1), res['perc_spec'].squeeze(1)
    
    mel = spec_to_mel_torch(
      spec, 
      hps.data.filter_length, 
      hps.data.n_mel_channels, 
      hps.data.sampling_rate,
      hps.data.mel_fmin, 
      hps.data.mel_fmax)
    
    # #V5,V6_H
    mel_H = spec_to_mel_torch(
      spec_H, 
      hps.data.filter_length, 
      hps.data.n_mel_channels, 
      hps.data.sampling_rate,
      hps.data.mel_fmin, 
      hps.data.mel_fmax)
    #V5_3
    
    # mel_P = spec_to_mel_torch(
    #   spec_P, 
    #   hps.data.filter_length, 
    #   hps.data.n_mel_channels, 
    #   hps.data.sampling_rate,
    #   hps.data.mel_fmin, 
    #   hps.data.mel_fmax)    

    
    # utils.plot_spectrogram_to_numpy(mel_P[0].data.cpu().numpy(), name = 'mel_P_T', fig_size=(4,8))
    # utils.plot_spectrogram_to_numpy(librosa.amplitude_to_db(spec[0].data.cpu().numpy()), name = 'spec_T', fig_size=(4,8))
    # utils.plot_spectrogram_to_numpy(mel_H[0].data.cpu().numpy(), name = 'mel_H_T', fig_size=(4,8))
    # utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy(), name = 'mel_org_T', fig_size=(4,8))

    
    ####
    
    with autocast(enabled=hps.train.fp16_run):
      # Generator -> y_hat
      # g: None, mel: None
      y_hat, ids_slice, (emb, emb_quantized, perplexity), (g, g_H) = net_g(c, spec, g=g, mels=[mel,mel_H])
      
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length) # mel 길이 28로 고정 slice
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )
      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 
      
      vq_spk_emb = emb - emb_quantized
      spk_emb = g
      
      # Discriminator loss
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        # LS-GAN loss
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    # Generator loss
    with autocast(enabled=hps.train.fp16_run):
      # Generator, fmap_r: discriminator의 layer 별 output for y, fmap_g: for y_hat
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = torch.tensor([0]).cuda(rank)
        loss_fm = feature_loss(fmap_r, fmap_g) # layer outputsA 간의 거리 계산 (L1 loss)
        loss_gen, losses_gen = generator_loss(y_d_hat_g) #LS-GAN loss
        loss_commitment, loss_codebook = vq_loss(emb, emb_quantized, commitment_labmda=0.25, codebook_labmda=1) #VQ_loss
        # loss_contrastive = contrastive_loss(vq_spk_emb, spk_emb) #Contrastive loss
        loss_contrastive = torch.tensor([0]).cuda(rank)
        loss_spk_M_and_H = F.l1_loss(g_H.detach(), g)
        loss_spk_VQ_and_H = F.l1_loss(g.detach().expand(vq_spk_emb.size()), vq_spk_emb)
        #이거 차원 맞나 확인하기
        
        loss_gen_all = loss_gen + loss_fm + loss_mel + 2*loss_commitment + 2*loss_codebook + loss_spk_M_and_H + loss_spk_VQ_and_H

        
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_commitment, loss_codebook, loss_spk_M_and_H, loss_spk_VQ_and_H, perplexity]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        if hps.setting.log_wandb:
          wandb.log({
            "loss/g_total": loss_gen_all.detach().cpu().numpy(),
            "loss/d_total": loss_disc_all.detach().cpu().numpy(),
            "learning_rate": lr,
            "grad_norm_d": grad_norm_d,
            "grad_norm_g": grad_norm_g,
            "loss/g_fm": loss_fm.detach().cpu().numpy(),
            "loss/g_mel": loss_mel.detach().cpu().numpy(),
            "loss/g_kl": loss_kl.detach().cpu().numpy(),
            "loss/g_comm": loss_commitment.detach().cpu().numpy(),
            "loss/g_code": loss_codebook.detach().cpu().numpy(),
            "loss/g_cont": loss_contrastive.detach().cpu().numpy(),
            "perplexity": perplexity.detach().cpu().numpy(),
            "train/org_mel": wandb.Image(y_mel[0].detach().cpu().numpy()),
            "train/gen_mel": wandb.Image(y_hat_mel[0].detach().cpu().numpy()),
            "train/gt_mel": wandb.Image(mel[0].detach().cpu().numpy()),
            # "train/gt_mel_P": wandb.Image(mel_P[0].detach().cpu().numpy()),
            "train/gt_mel_H": wandb.Image(mel_H[0].detach().cpu().numpy()),
            
          })
    
      # default: 10000 stpe 마다 
      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, [net_g, net_d], eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
# def evaluate(hps, generator, eval_loader, writer_eval):
def evaluate(hps, nets, eval_loader, writer_eval):

    generator, net_d = nets
    
    generator.eval()
    
    losses_list = [0, 0, 0, 0]
    hpss_module = torch_hpss.HPSS(kernel_size=31, reduce_method='median').cuda(0)
    
    with torch.no_grad():
      for batch_idx, items in enumerate(eval_loader):
        print('Evaluate GPU')
        # GPUtil.showUtilization()
        
        if hps.model.use_spk:
          c, spec, y, spk = items
          # g = spk[:1].cuda(0)
          g = spk.cuda(0)

        else:
          c, spec, y = items
          g = None
          break
        # spec, y = spec[:1].cuda(0), y[:1].cuda(0)
        # c = c[:1].cuda(0)
      spec, y = spec.cuda(0), y.cuda(0)
      #V5_3
      c = spec
      
      res = hpss_module(spec.unsqueeze(1))
      spec_H, spec_P = res['harm_spec'].squeeze(1), res['perc_spec'].squeeze(1)
      
      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      # mel_H = spec_to_mel_torch(
      #   spec_H, 
      #   hps.data.filter_length, 
      #   hps.data.n_mel_channels, 
      #   hps.data.sampling_rate,
      #   hps.data.mel_fmin, 
      #   hps.data.mel_fmax) 
      

      y_hat = generator.module.infer(c, g=g, mels=mel)
      
      
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
      
      # Discriminator loss
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        # LS-GAN loss
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
        losses_list[0] += loss_disc_all
        
      with autocast(enabled=hps.train.fp16_run):
      # Generator, fmap_r: discriminator의 layer 별 output for y, fmap_g: for y_hat
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
        with autocast(enabled=False):
          loss_mel = F.l1_loss(mel, y_hat_mel) * hps.train.c_mel
          # loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
          loss_fm = feature_loss(fmap_r, fmap_g) # layer outputsA 간의 거리 계산 (L1 loss)
          loss_gen, losses_gen = generator_loss(y_d_hat_g) #LS-GAN loss
          loss_gen_all = loss_gen + loss_fm + loss_mel # kl loss 뺌 (infer)
          losses_list[1] += loss_gen_all
          losses_list[2] += loss_fm
          losses_list[3] += loss_mel
    
    if hps.setting.log_wandb:
      wandb.log({
            "eval_loss/d_total": losses_list[0].detach().cpu().numpy(),
            "eval_loss/g_total": losses_list[1].detach().cpu().numpy(),
            "eval_loss/g_fm": losses_list[2].detach().cpu().numpy(),
            "eval_loss/g_mel": losses_list[3].detach().cpu().numpy(),
            "eval/gen_mel": wandb.Image(y_hat_mel[0].detach().cpu().numpy()),
            "eval/gt_mel": wandb.Image(mel[0].detach().cpu().numpy()),
            "eval/gt_mel_P": wandb.Image(c[0].detach().cpu().numpy()),
            # "eval/gt_mel_H": wandb.Image(mel_H[0].detach().cpu().numpy()),
            
      })
    
    generator.train()

                           
if __name__ == "__main__":
  main()
