
import sys, os
import torch
sys.path.append('/home/sim/VoiceConversion/FreeVC')

from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
  vq_loss,
  info_nce_loss
)

from torch.nn import functional as F




a = torch.tensor([[1,2],[3,4],[5,6]], dtype=float).unsqueeze(0)
a = a.expand(3, -1, -1)
a = a.unsqueeze(0)
a = a.expand(4, -1, -1, 2)
a = a.permute((2,1,0,3))
print('o')


b = torch.tensor([[[1, 2],
         [3, 4],
         [5, 6],
         [7, 8]],

        [[10, 20],
         [30, 40],
         [50, 60],
         [70, 80]],

        [[100, 200],
         [300, 400],
         [500, 600],
         [700, 800]]], dtype=float)


b = b.unsqueeze(0)
b = b.expand(3, -1, -1, -1)


# s_q = torch.rand(1, 265, 28)
# s_e = torch.rand(64, 256, 1)

cos_sim = F.cosine_similarity(a, b, dim=-1)
con_sim = F.sum(cos_sim, dim=-1)
# cos_sim eye 가 positive, 나머지는 negetive

print(cos_sim)
