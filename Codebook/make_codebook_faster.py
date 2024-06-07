#%
import numpy as np
import faiss

class FaissKMeans:
    def __init__(self, n_clusters=10, n_init=100, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X, y):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]
    

import faiss
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
  
# flag=0
# for path in ref_paths_new :
#     tmp = torch.load(path)
#     tmp = tmp.squeeze().transpose(0,1)
    

#     if flag == 0:
#         total = tmp
#         flag = 1
#     else:
#         total = torch.concat((total, tmp), dim=0)
        
# # Perform t-SNE
# tsne = TSNE(n_components=3, random_state=42)
# tsne_results = tsne.fit_transform(total)
# #%
# # Plot t-SNE results in 3D
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot
# sc = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c='blue', alpha=0.6, s=0.1)

# # Labels and title
# ax.set_title('3D t-SNE visualization')
# ax.set_xlabel('t-SNE component 1')
# ax.set_ylabel('t-SNE component 2')
# ax.set_zlabel('t-SNE component 3')

# plt.show()
# plt.savefig('./tsne_data.png')
# total = total / (torch.norm(total, dim=1, keepdim=True) + 1e-4)

#%

sorted_list_1 = sorted(ref_paths_new)
sorted_list_2 =  sorted(src_paths_new)
flag = 0
for path_ref, path_src in zip(sorted_list_1, sorted_list_2):
    t1 = torch.load(path_ref).squeeze().transpose(0,1)
    t2 = torch.load(path_src).squeeze().transpose(0,1)
    
    tmp = torch.concat((t1, t2), dim=0)
    

    if flag == 0:
        total = tmp
        flag = 1
    else:
        total = torch.concat((total, tmp), dim=0)

# # Perform t-SNE
# tsne = TSNE(n_components=3, random_state=42)
# tsne_results = tsne.fit_transform(total)

# # Plot t-SNE results in 3D
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot
# sc = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c='blue', alpha=0.6, s=0.1)

# # Labels and title
# ax.set_title('3D t-SNE visualization')
# ax.set_xlabel('t-SNE component 1')
# ax.set_ylabel('t-SNE component 2')
# ax.set_zlabel('t-SNE component 3')

# plt.show()
# plt.savefig('./tsne_data.png')      



#%
from sklearn.preprocessing import StandardScaler
data = total.numpy().astype(np.float32)
# scaler = StandardScaler()
# data = scaler.fit_transform(data)
# Setup

kmeans = faiss.Kmeans(d=1024, k=2048, niter=70, verbose=True)
# For GPU(s), run the following line. This will use all GPUs
# kmeans = faiss.Kmeans(d=D, k=K, niter=20, verbose=True, gpu=True)

# Run clustering
kmeans.train(data)

# Error for each iteration
print(kmeans.obj)  # array with 20 elements

# Centroids after clustering
print(kmeans.centroids.shape)  # (10, 128)

# The assignment for each vector.
dists, ids = kmeans.index.search(data, 1)  # Need to run NN search again
print(ids.shape)  # (10000, 1)

# Params
print("D:", kmeans.d)
print("K:", kmeans.k)
print("niter:", kmeans.cp.niter)
#%
all = np.concatenate((data, kmeans.centroids))
# Perform t-SNE
tsne = TSNE(n_components=3, random_state=42)
tsne_results = tsne.fit_transform(all)
#%


#%
# Plot t-SNE results in 3D
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(tsne_results[:data.shape[0], 0], tsne_results[:data.shape[0], 1], tsne_results[:data.shape[0], 2], c='blue', alpha=0.3, s=0.1)
# Scatter plot
ax.scatter(tsne_results[data.shape[0]:, 0], tsne_results[data.shape[0]:, 1], tsne_results[data.shape[0]:, 2], c='red', alpha=1, s=1)

# Labels and title
ax.set_title('3D t-SNE visualization')
ax.set_xlabel('t-SNE component 1')
ax.set_ylabel('t-SNE component 2')
ax.set_zlabel('t-SNE component 3')

plt.show()
plt.savefig('./tsne_codebook.png')   
#%
torch.save(torch.from_numpy(kmeans.centroids), '/shared/racoon_fast/sim/codebook_init/codebook_2048_SrcRef.pt')
# np.save(kmeans.centroids), '/shared/racoon_fast/sim/codebook_init/codebook_2048_SrcRef.pt')
#%
kmeans.centroids.shape
