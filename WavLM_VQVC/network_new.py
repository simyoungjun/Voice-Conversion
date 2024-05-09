import torch
import torch.nn as nn
import torch.nn.functional as F
import module as mm
import numpy as np
from config_new import Arguments as args

class Encoder(nn.Module):
	"""
	Encoder
		Args:
			mel: (N, Tx, C_mel) log-melspectrogram (variable length)
		Returns:
			y_: (N, Tx, C_hidden)
	"""

	def __init__(self, mel_channels=1024, z_dim=256):
		super(Encoder, self).__init__()
		self.encoder = nn.Sequential(
			mm.Conv1d(mel_channels, 64, kernel_size=4, padding='same', bias=True, activation_fn=nn.ReLU),
			mm.Conv1d(64, 256, kernel_size=3, padding='same', bias=True),
			mm.Conv1d(256, 128, kernel_size=3, padding='same', bias=True),
			mm.Conv1dResBlock(128, 128, kernel_size=3, padding='same', bias=True, activation_fn=nn.ReLU),
			mm.Conv1dResBlock(128, 128, kernel_size=3, padding='same', bias=True, activation_fn=nn.ReLU),
			mm.Conv1d(128, z_dim, kernel_size=1, padding='same', bias=False)
		)

	def forward(self, mels):
		z = self.encoder(mels)
		return z


class VQEmbeddingEMA(nn.Module):
	"""
		VQEmbeddingEMA
			- vector quantization module
			- ref
				from VectorQuantizedCPC official repository
				(https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py)

		encode:
			args:
				x:	(N, T, z_dim)
			returns:
				quantized:	(N, T, z_dim)
				indices:	(N, T)
		forward:
			args:
				x:	(N, T, z_dim)
			returns:
				quantized: (N, T, z_dim)
				loss:	(N, 1)
				perplexity: (N, 1)
	"""


	def __init__(self, n_embeddings, embedding_dim, epsilon=1e-5):
		super(VQEmbeddingEMA, self).__init__()
		self.epsilon = epsilon

		# init_bound = 1 / n_embeddings
		# embedding = torch.Tensor(n_embeddings, embedding_dim)
		# embedding.uniform_(-init_bound, init_bound)
		# embedding = embedding / (torch.norm(embedding, dim=1, keepdim=True) + 1e-4)
		# embedding = torch.load('/shared/racoon_fast/sim/codebook_init/codebook.pt').permute(1,0)
		embedding = torch.from_numpy(np.load('/shared/racoon_fast/sim/codebook_init/codebook.npy'))
  
		self.register_buffer("embedding", embedding)
		self.register_buffer("ema_count", torch.zeros(n_embeddings))
		self.register_buffer("ema_weight", self.embedding.clone())

	def instance_norm(self, x, dim, epsilon=1e-5):
		mu = torch.mean(x, dim=dim, keepdim=True)
		std = torch.std(x, dim=dim, keepdim=True)

		z = (x - mu) / (std + epsilon)
		return z

	def cosine_sim(self, x, codebook): # X: (batch, T, z_dim) , codebook: (codebook_size, z_dim)
		M, D = codebook.size()
		x_c = x.unsqueeze(2).expand(-1, -1, 256, -1)
		codebook_c = codebook.unsqueeze(0).unsqueeze(0).expand(x.size(0) , x_c.size(1), -1, -1)
		# codebook = codebook.expand(64, -1, -1, -1)

		cos_sim = F.cosine_similarity(x_c, codebook_c, dim=-1)
		cos_min = 1 - cos_sim 
		indices = torch.argmin(cos_min.float(), dim=-1)
		quantized = F.embedding(indices, codebook)
		encodings = F.one_hot(indices, M).float()
  
		return quantized, encodings

	def L2_distance(self, x, embedding): # X: (batch, T, z_dim) , codebook: (codebook_size, z_dim)
		embedding_norm = embedding / (torch.norm(embedding, dim=1, keepdim=True) + 1e-4)

		M, D = embedding_norm.size()
		x_flat = x.detach().reshape(-1, D)

		distances = torch.addmm(torch.sum(embedding_norm ** 2, dim=1) +
								torch.sum(x_flat ** 2, dim=1, keepdim=True),
								x_flat, embedding_norm.t(),
								alpha=-2.0, beta=1.0)

		indices = torch.argmin(distances.float(), dim=-1).detach()
		encodings = F.one_hot(indices, M).float()
		quantized = F.embedding(indices, embedding)

		quantized = quantized.view_as(x)
		return quantized, encodings

	def forward(self, x):

		# x = self.instance_norm(x, dim=1)

		# embedding = self.embedding / (torch.norm(self.embedding, dim=1, keepdim=True) + 1e-4)
		codebook = self.embedding

		if x.size(0) != 64:
			print(x.size(0) )
		# cosine similarity metric
		quantized, encodings = self.L2_distance(x, codebook)
		# quantized, encodings = self.cosine_sim(x,codebook)

		commitment_loss = F.mse_loss(x.detach(), quantized.detach())

		quantized_ = x + (quantized - x).detach()
		quantized_ = (quantized_ + quantized)/2

		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

		import matplotlib.pyplot as plt
		# spk embedding correlation 확인
		spk = (x - quantized_).reshape(-1, 1024)
  
		# # cosine metric
		# spk_1 = spk.unsqueeze(0).expand(spk.size(0), -1, -1)
		# spk_2 = spk.unsqueeze(1).expand(-1, spk.size(0), -1)
		# cos_sim = F.cosine_similarity(spk_1, spk_2, dim=-1)
		# heatmap = plt.imshow(cos_sim.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')
		# L2
		spk_1 = spk.unsqueeze(0)
		spk_2 = spk.unsqueeze(0)
		L2_dist=torch.cdist(spk_1, spk_2).squeeze()
		# Create the plot
		plt.figure(figsize=(8, 6))
		heatmap = plt.imshow(L2_dist.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')
  

		# Add a color bar
		plt.colorbar(heatmap)

		# Add title and labels as needed
		plt.title('2D Array Heat Map')
		plt.xlabel('X-axis Label')
		plt.ylabel('Y-axis Label')
		# plt.savefig(f'eval_fig/spk_emb/{args.model_name}_cos.png')
		plt.savefig(f'eval_fig/spk_emb/{args.model_name}_L2.png')
		# Show the plot
  
		return quantized_, commitment_loss, perplexity



class Decoder(nn.Module):
	"""
		Decoder

		args:
			z_enc:	(N, T, z_dim)
			z_quan:	(N, T, z_dim)
		return:
			mel_reconstructed:	(N, T, C_mel)	
	"""

	def __init__(self, in_channels=1024, mel_channels=80):
		super(Decoder, self).__init__()

		self.res_blocks = nn.Sequential(
					mm.Conv1d(in_channels, 512, kernel_size=5, 
								bias=False, padding='same'),
					mm.Conv1d(512, 128, kernel_size=3, 
								bias=False, padding='same'),
					mm.Conv1dResBlock(128, 128, kernel_size=3, 
								bias=True, padding='same', activation_fn=nn.ReLU),
					mm.Conv1dResBlock(128, 128, kernel_size=3, 
								bias=True, padding='same', activation_fn=nn.ReLU),
					# mm.Upsample(scale_factor=2, mode='nearest'),
					mm.Conv1d(128, 256, kernel_size=2, 
								bias=True, padding='same', activation_fn=nn.ReLU),
					mm.Linear(256, mel_channels)					
		)

		# spk_emb dim 256 -> z_dim 으로 바꿔줌 나중에 더해주기 위함
		self.cond = nn.Conv1d(1024, 64, 1)
        
	def forward(self, contents, speaker_emb):

		# contents = self.norm(contents, dim=2)
		# speaker_emb = self.norm(speaker_emb, dim=2)

		embedding = contents + speaker_emb

		mel_reconstructed = self.res_blocks(embedding)

		return mel_reconstructed


	# def evaluate(self, src_contents, speaker_emb, speaker_emb_):

	# 	# normalize the L2-norm of input  vector into 1 on every time-step
	# 	src_contents = self.norm(src_contents, dim=2)
	# 	speaker_emb = self.norm(speaker_emb, dim=2)
	# 	speaker_emb_ = self.norm(speaker_emb_, dim=2)

	# 	embedding = src_contents + speaker_emb

	# 	# converted mel_hat 
	# 	mel_converted = self.res_blocks(embedding)

	# 	# only src-code
	# 	mel_src_code = self.res_blocks(src_contents)

	# 	# only ref-style
	# 	mel_ref_style = self.res_blocks(speaker_emb_)

	# 	return mel_converted, mel_src_code, mel_ref_style

	# def convert(self, src_contents, src_style_emb_, ref_contents, ref_speaker_emb, ref_speaker_emb_):
	# 	# normalize the L2-norm of input  vector into 1 on every time-step
	# 	src_contents = self.norm(src_contents, dim=2)
	# 	src_style_emb_ = self.norm(src_style_emb_, dim=2)
	# 	ref_contents = self.norm(ref_contents, dim=2)
	# 	ref_speaker_emb = self.norm(ref_speaker_emb, dim=2)
	# 	ref_speaker_emb_ = self.norm(ref_speaker_emb_, dim=2)

	# 	embedding = src_contents + ref_speaker_emb

	# 	# converted mel_hat 
	# 	mel_converted = self.res_blocks(embedding)

	# 	# only src-code
	# 	mel_src_code = self.res_blocks(src_contents)

	# 	# only src_style_emb_
	# 	mel_src_style = self.res_blocks(src_style_emb_)

	# 	# only ref-code
	# 	mel_ref_code = self.res_blocks(ref_contents)	

	# 	# only ref_style_emb_
	# 	mel_ref_style = self.res_blocks(ref_speaker_emb_)

	# 	return mel_converted, mel_src_code, mel_src_style, mel_ref_code, mel_ref_style




	def norm(self, x, dim, epsilon=1e-4):
		x_ = x / (torch.norm(x, dim=dim, keepdim = True) + epsilon)
		return x_


