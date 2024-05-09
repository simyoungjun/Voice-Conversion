import sys
sys.path.append("/home/monet/VQVC")

from config_new import Arguments as args

from utils.vocoder import vocgan_infer
from utils.path import create_dir, get_path
from utils.dataset import de_normalize
from utils.figure import draw_melspectrogram
import os
import sys
sys.path.append("/home/monet/VQVC/FreeVC-main")
from mel_processing import spectrogram_torch, spec_to_mel_torch

import torch

def evaluate(model, vocoder, eval_data_loader, criterion, global_step, mel_stat, writer=None, DEVICE=None, hps=None):

	model.eval()
	# mel_mean, mel_std = mel_stat 

	with torch.no_grad():
		eval_loss, eval_recon_L1_loss, eval_recon_L2_loss, eval_perplexity, eval_commitment_loss = 0, 0, 0, 0, 0

		for step, items in enumerate(eval_data_loader):

			c, spec, y= items

			c, spec= c.to(DEVICE), spec.to(DEVICE)
			
			mels = spec_to_mel_torch(
				spec, 
				hps.data.filter_length, 
				hps.data.n_mel_channels, 
				hps.data.sampling_rate,
				hps.data.mel_fmin, 
				hps.data.mel_fmax)

			mels = mels.permute(0,2,1)
			c = c.permute(0,2,1)
			# mels_hat, mels_code, mels_style, commitment_loss, perplexity = model.evaluate(c.detach(), spk.detach())
			mels_hat, commitment_loss, perplexity = model(c.detach())
   

			commitment_loss = args.commitment_cost * commitment_loss
			recon_L1_loss = criterion(mels, mels_hat)

			total_loss = commitment_loss + recon_L1_loss

			eval_perplexity += perplexity.item()
			eval_recon_L1_loss += recon_L1_loss.item()
			# eval_recon_L2_loss += recon_L2_loss.item()
   
			eval_commitment_loss += commitment_loss.item()
			eval_loss += total_loss.item()

		print(f'step: {global_step}, eval_all_loss: {eval_loss / len(eval_data_loader)}, eval_recon_L1_loss: {eval_recon_L1_loss/len(eval_data_loader)}, eval_commit_loss: {commitment_loss/len(eval_data_loader)}')
			
		# mel = de_normalize(mels[0], mean=mel_mean, std=mel_std).float()
		# mel_hat = de_normalize(mels_hat[0], mean=mel_mean, std=mel_std).float()
		# mel_code = de_normalize(mels_code[0], mean=mel_mean, std=mel_std).float()
		# mel_style = de_normalize(mels_style[0], mean=mel_mean, std=mel_std).float()
		mel = mels[0]
		mel_hat = mels_hat[0]
		vocgan_infer(mel.transpose(0, 1), vocoder, path=get_path(args.eval_path, "{:0>3}_GT.wav".format(global_step//1000)))
		vocgan_infer(mel_hat.transpose(0, 1), vocoder, path=get_path(args.eval_path, "{:0>3}_reconstructed.wav".format(global_step//1000)))
		# vocgan_infer(mel_code.transpose(0, 1), vocoder, path=get_path(args.eval_path, "{:0>3}_code.wav".format(global_step//1000)))
		# vocgan_infer(mel_style.transpose(0, 1), vocoder, path=get_path(args.eval_path, "{:0>3}_style.wav".format(global_step//1000)))

		mel =  mel.view(-1, 80).detach().cpu().numpy().T
		mel_hat = mel_hat.view(-1, 80).detach().cpu().numpy().T
		# mel_code = mel_code.view(-1, 80).detach().cpu().numpy().T
		# mel_style = mel_style.view(-1, 80).detach().cpu().numpy().T
  
		# fig = draw_melspectrogram(mel, mel_hat, mel_code, mel_style)
		fig = draw_melspectrogram(mel, mel_hat, None, None)
		directory_path = f'./eval_fig/{args.model_name}'
		if not os.path.exists(directory_path):
			os.makedirs(directory_path)
		fig.savefig(f'{directory_path}/{global_step}.png')	
		# if args.log_tensorboard:
		# 	writer.add_scalars(mode="eval_reconstruction_loss", global_step=global_step, loss=eval_recon_loss / len(eval_data_loader))
		# 	writer.add_scalars(mode="eval_commitment_loss", global_step=global_step, loss=eval_commitment_loss / len(eval_data_loader))
		# 	writer.add_scalars(mode="eval_perplexity", global_step=global_step, loss=eval_perplexity / len(eval_data_loader))
		# 	writer.add_scalars(mode="eval_total_loss", global_step=global_step, loss=eval_loss / len(eval_data_loader))
		# 	writer.add_mel_figures(mode="eval-mels_", global_step=global_step, mel=mel, mel_hat=mel_hat, mel_code=mel_code, mel_style=mel_style)


		

	
