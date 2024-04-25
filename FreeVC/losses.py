import torch 
from torch.nn import functional as F

import commons


def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()
  #print(logs_p)
  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l

def vq_loss(x, quantized, commitment_labmda=0.25, codebook_labmda=1, posterior_emb=None):
  #detach()로 codebook 은 gradient update하지 않고, encoder만 codebook에 가까울 수 있게 update
  # codebook loss 없는 버전
  encoder_latent_loss = F.mse_loss(x, quantized.detach())
  commitment_loss = commitment_labmda * encoder_latent_loss
  
  # codebook loss 있는 버전
  if codebook_labmda == 0:
    codebook_loss = torch.tensor([0])
  else:
    if posterior_emb == None:
      codebook_loss = F.mse_loss(x.detach(), quantized) #latent_loss
    else:
      codebook_loss = F.mse_loss(posterior_emb.detach(), quantized)
      
  return commitment_loss, codebook_loss



def info_nce_loss(emb, g, mode='train'):
    imgs, _ = emb, g
    imgs = torch.cat(imgs, dim=0)

    # Encode all images
    # feats = self.convnet(imgs)
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / self.hparams.temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    # Logging loss
    self.log(mode+'_loss', nll)
    # Get ranking position of positive example
    comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                          cos_sim.masked_fill(pos_mask, -9e15)],
                          dim=-1)
    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    # Logging ranking metrics
    self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
    self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
    self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

    return nll
