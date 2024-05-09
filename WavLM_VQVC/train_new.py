from config_new import Arguments as args
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import sys, random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from model_new import VQVC

from evaluate_new import evaluate
# from dataset_new import SpeechDataset #, collate_fn

from utils.scheduler import WarmupScheduler
from utils.checkpoint import load_checkpoint, save_checkpoint
# from utils.writer import Writer
from utils.vocoder import get_vocgan

from tqdm import tqdm


sys.path.append("/home/sim/VoiceConversion/FreeVC")
import commons
import utils_freevc
from dataset_new import (
	TextAudioSpeakerLoader,
	TextAudioSpeakerCollate,
	DistributedBucketSampler
)
from mel_processing import spectrogram_torch, spec_to_mel_torch
# import wandb


hps = utils_freevc.get_hparams(args = args)


segment_size=40

def train(train_data_loader, eval_data_loader, model, reconstruction_loss, vocoder, mel_stat, optimizer, scheduler, global_step, writer=None, DEVICE=None):

	model.train()

	while global_step < args.max_training_step:

		for step, items in tqdm(enumerate(train_data_loader), total=len(train_data_loader), unit='B', ncols=70, leave=False):
			c, spec, y = items

			c, spec= c.to(DEVICE), spec.to(DEVICE)
			
			mels = spec_to_mel_torch(
				spec, 
				hps.data.filter_length, 
				hps.data.n_mel_channels, 
				hps.data.sampling_rate,
				hps.data.mel_fmin, 
				hps.data.mel_fmax)

			# T_mel = mels.size(2)

			# while T_mel <= 40:
			# 	mels = torch.cat((mels, mels), dim=2)

			# mels_slice, ids_slice = commons.rand_slice_segments(mels, segment_size)
			# c_slice = commons.slice_segments(c, ids_slice, segment_size)
			# y_slice = commons.slice_segments(y, ids_slice, segment_size * hps.data.hop_length)

			# mels_slice = mels_slice.permute(0,2,1)
			# optimizer.zero_grad()

			# mels_hat, commitment_loss, perplexity = model(mels_slice.detach())

			mels = mels.permute(0,2,1)
			c = c.permute(0,2,1)
			
			optimizer.zero_grad()
			
			# 여기에 mels 대신에 c 넣기
			mels_hat, commitment_loss, perplexity = model(c.detach())

			commitment_loss = args.commitment_cost * commitment_loss
			recon_loss = reconstruction_loss(mels_hat, mels) 

			loss = commitment_loss + recon_loss 
			loss.backward()
			# wandb.log({
			# 	"Train Loss": loss.cpu().detach().npy(),

			# 	})

			nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)
			optimizer.step()

			if global_step % args.save_checkpoint_step == 0:
				save_checkpoint(checkpoint_path=args.model_checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler, global_step=global_step)

			if global_step % args.eval_step == 0:
				print(f'\nstep: {global_step}, train_all_loss: {loss}, recon_loss: {recon_loss}')
				evaluate(model=model, vocoder=vocoder, eval_data_loader=eval_data_loader, criterion=reconstruction_loss, mel_stat=mel_stat, global_step=global_step, writer=writer, DEVICE=DEVICE, hps=hps)
				model.train()

			# if args.log_tensorboard:
			# 	writer.add_scalars(mode="train_recon_loss", global_step=global_step, loss=recon_loss)
			# 	writer.add_scalars(mode="train_commitment_loss", global_step=global_step, loss=commitment_loss)
			# 	writer.add_scalars(mode="train_perplexity", global_step=global_step, loss=perplexity)
			# 	writer.add_scalars(mode="train_total_loss", global_step=global_step, loss=loss)

			global_step += 1

		scheduler.step()

def main(DEVICE):
	# wandb.init('Chaehyeon')
	# define model, optimizer, scheduler
	model = VQVC().to(DEVICE)

	recon_loss = nn.L1Loss().to(DEVICE)
	vocoder = get_vocgan(ckpt_path=args.vocoder_pretrained_model_path).to(DEVICE)

	# mel_stat = torch.tensor(np.load(args.mel_stat_path)).to(DEVICE)

	optimizer = Adam(model.parameters(), lr=args.init_lr)
	scheduler = WarmupScheduler( optimizer, warmup_epochs=args.warmup_steps,
        			initial_lr=args.init_lr, max_lr=args.max_lr,
				milestones=args.milestones, gamma=args.gamma)

	global_step = load_checkpoint(checkpoint_path=args.model_checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler)

	
	train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
	train_sampler = DistributedBucketSampler(
		train_dataset,
		hps.train.batch_size,
		[32,300,400,500,600,700,800,900,1000],
		num_replicas=1,
		rank=0,
		shuffle=True)
	collate_fn = TextAudioSpeakerCollate(hps)
	train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
		collate_fn=collate_fn, batch_sampler=train_sampler)

	eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
	eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=True,
		batch_size=hps.train.batch_size, pin_memory=False,
		drop_last=False, collate_fn=collate_fn)
	# # load dataset & dataloader
	# train_dataset = SpeechDataset(mem_mode=args.mem_mode, meta_dir=args.prepro_meta_train, dataset_name = args.dataset_name, mel_stat_path=args.mel_stat_path, max_frame_length=args.max_frame_length)
	# eval_dataset = SpeechDataset(mem_mode=args.mem_mode, meta_dir=args.prepro_meta_eval, dataset_name=args.dataset_name, mel_stat_path=args.mel_stat_path, max_frame_length=args.max_frame_length)
	mel_stat = None
	# train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.n_workers)
	# eval_data_loader = DataLoader(dataset=eval_dataset, batch_size=args.train_batch_size, shuffle=False, pin_memory=True, drop_last=True)

	# tensorboard
	# writer = Writer(args.model_log_path) if args.log_tensorboard else None		
	writer=None
	# train the model!
	train(train_loader, eval_loader, model, recon_loss, vocoder, mel_stat, optimizer, scheduler, global_step, writer, DEVICE)

# 데이터셋 로딩 테스트 부분 추가
# def test_dataset_loading():
#     print("Testing dataset loading...")
#     dataset = SpeechDataset(mem_mode=args.mem_mode, meta_dir=args.prepro_meta_train, dataset_name=args.dataset_name, mel_stat_path=args.mel_stat_path, max_frame_length=args.max_frame_length)
#     print(f"Dataset size: {len(dataset)}")
#     if len(dataset) > 0:
#         for i in range(min(5, len(dataset))):  # 처음 5개 샘플을 로드하여 테스트
#             sample, _ = dataset[i]
#             print(f"Sample {i}: Shape {sample.shape}")
#     else:
#         print("No data found in the dataset.")

if __name__ == "__main__":

	print("[LOG] Start training...")
	print()
	DEVICE = torch.device("cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu")

 
	# test_dataset_loading()  # 데이터셋 로딩 테스트 수행
	main(DEVICE)	
