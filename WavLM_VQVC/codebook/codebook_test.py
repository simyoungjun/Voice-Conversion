#%
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


c = np.load('/shared/racoon_fast/sim/codebook_init/codebook.npy')
#%
# Make Codebook
titles, srcs, tgts = [], [], []
with open('/home/sim/VoiceConversion/conversion_metas/VCTK_seen_pairs.txt', "r") as file:
    for rawline in file.readlines()[:64]:
        title, tgt, src = rawline.strip().split("|")

        titles.append('src:'+src.split('/')[-1][:-4]+'&tgt:'+tgt.split('/')[-1][:-4])
        
        
        srcs.append(src)
        tgts.append(tgt)

        substring_to_replace1 = 'vctk-16k'
        replacement_string1 = 'wavlm-6L' 
        substring_to_replace2 = '.wav'
        replacement_string2 = '.pt' 
        
        ref_paths_new = [ref_path.replace(substring_to_replace1, replacement_string1).replace(substring_to_replace2, replacement_string2) for ref_path in srcs]
        src_paths_new = [src_path.replace(substring_to_replace1, replacement_string1).replace(substring_to_replace2, replacement_string2) for src_path in tgts]
        
flag=0
for path in ref_paths_new[:] :
    tmp = torch.load(path)[:,:,:24]
    if flag == 0:
        total = tmp
        flag = 1
    else:
        total = torch.concat((total, tmp), dim=0)

#%
# total = total / (torch.norm(total, dim=1, keepdim=True) + 1e-4)
batch = total.permute(0,2,1)

codebook = torch.load('/shared/racoon_fast/sim/codebook_init/codebook.pt')

source_norms = torch.norm(batch, p=2, dim=-1)
matching_norms = torch.norm(codebook, p=2, dim=-1)
batch = batch.unsqueeze(2).expand(-1, -1, 256, -1)
codebook_c = codebook.unsqueeze(0).unsqueeze(0).expand(64, 24, -1, -1)
# codebook = codebook.expand(64, -1, -1, -1)

cos_sim = F.cosine_similarity(batch, codebook_c, dim=-1)
cos_min = 1 - cos_sim 
indices = torch.argmin(cos_min.float(), dim=-1).detach()
quantized = F.embedding(indices, codebook)


dotprod = -torch.cdist(batch[None], codebook[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
dotprod /= 2

dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )


# distances = torch.addmm(torch.sum(embedding ** 2, dim=1) +
#                         torch.sum(x_flat ** 2, dim=1, keepdim=True),
#                         x_flat, embedding.t(),
#                         alpha=-2.0, beta=1.0)