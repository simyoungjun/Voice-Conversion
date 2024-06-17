
#%
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import torch

# n_embeddings = 256
# embedding_dim = 32

# init_bound = 1 / n_embeddings
# embedding = torch.Tensor(n_embeddings, embedding_dim)
# embedding.uniform_(-init_bound, init_bound)
# print(embedding)
# embedding = embedding / (torch.norm(embedding, dim=1, keepdim=True) + 1e-4)

# #%
# kmeans = KMeans(n_clusters=40, random_state=1)
# df = pd.DataFrame(embedding.numpy())
# print(df)

# kmeans.fit(df)


#%
# Make Codebook
titles, srcs, tgts = [], [], []
with open('/home/sim/VoiceConversion/conversion_metas/VCTK_seen_pairs.txt', "r") as file:
    for rawline in file.readlines()[:]:
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
for path in ref_paths_new :
    tmp = torch.load(path)
    tmp = tmp.squeeze().transpose(0,1)
    if flag == 0:
        total = tmp
        flag = 1
    else:
        total = torch.concat((total, tmp), dim=0)

# total = total / (torch.norm(total, dim=1, keepdim=True) + 1e-4)
df = df.DataFrame.from_pandas(pd.DataFrame(total.numpy()))

kmeans = KMeans(n_clusters=1024, random_state=42, n_init=3)
kmeans.fit(df)
#%
# codebook = kmeans.cluster_centers_
# np.save('/shared/racoon_fast/sim/codebook_init/codebook.npy', codebook)
# # torch.save(torch.from_numpy(kmeans.cluster_centers_), '/shared/racoon_fast/sim/codebook_init/codebook.pt')
#%

