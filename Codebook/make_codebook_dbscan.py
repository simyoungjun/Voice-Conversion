
#%
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
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
df = total.numpy()
#%

import numpy as np
from scipy.spatial.distance import euclidean, cityblock, pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

# Example two-dimensional array (each row is a vector)

# vector_magnitudes = np.linalg.norm(df, axis=-1)
# print(vector_magnitudes[:100])
# for i in range(20):
#     # 1. Euclidean Distance
#     euclidean_distance = euclidean(df[3], df[i])
#     print(euclidean_distance)
#%
import matplotlib.pyplot as plt
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(df)
# array = data_scaled
# vector_magnitudes = np.linalg.norm(data_scaled, axis=-1)
# print(vector_magnitudes[:100])

# for i in range(400):
#     # 1. Euclidean Distance
#     euclidean_distance = euclidean(data_scaled[3], data_scaled[i])
#     print(euclidean_distance)

from sklearn.neighbors import NearestNeighbors
# k-최근접 이웃 계산 (k는 min_samples로 설정)
k = 20  # min_samples 값
nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(df)
distances, indices = nbrs.kneighbors(df)

# k-거리 계산 (각 포인트에서 k번째 이웃까지의 거리)
k_distances = distances[:, -1]
k_distances = np.sort(k_distances)

# k-거리 그래프 그리기
plt.plot(k_distances)
plt.xlabel('Data Points sorted by distance')
plt.ylabel(f'{k}th Nearest Neighbor Distance')
plt.title(f'k-Distance Graph for k={k}')
plt.show()
#%
# Step 4: Apply DBSCAN
eps= 0.27
ms = 5
dbscan = DBSCAN(eps=eps, min_samples=ms, metric='cosine')
dbscan.fit(df)

# Step 5: Create the codebook
# Find unique labels (-1 is considered noise, so we exclude it)
unique_labels = set(dbscan.labels_)
if -1 in unique_labels:
    print(f'eps:{eps}, min_samples: {ms}')
    print('noise: ',len(dbscan.labels_[dbscan.labels_==-1]))
    print('cluster_num: ',len(unique_labels))
    # unique_labels.remove(-1)
#%
# Calculate the centroid of each cluster to form the codebook
codebook = []
cluster_points = []
for label in unique_labels:
    cluster_points = df[dbscan.labels_ == label]
    print(len(cluster_points))
    centroid = cluster_points.mean(axis=0)
    codebook.append(centroid)
# print(len(codebook))
#%

nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(df)
distances, indices = nbrs.kneighbors(cluster_points)

# k-거리 계산 (각 포인트에서 k번째 이웃까지의 거리)
k_distances = distances[:, -1]
k_distances = np.sort(k_distances)
# k-거리 그래프 그리기
plt.plot(k_distances)
plt.xlabel('Data Points sorted by distance')
plt.ylabel(f'{k}th Nearest Neighbor Distance')
plt.title(f'k-Distance Graph for k={k}')
plt.show()
eps, ms = 0.4, 5
dbscan = DBSCAN(eps=eps, min_samples=ms, metric='cosine')
dbscan.fit(cluster_points)
unique_labels = set(dbscan.labels_)
if -1 in unique_labels:
    print(f'eps:{eps}, min_samples: {ms}')
    print('noise: ',len(dbscan.labels_[dbscan.labels_==-1]))
    print('cluster_num: ',len(unique_labels))
    # unique_labels.remove(-1)
    
codebook = []
cluster_points = []
for label in unique_labels:
    cluster_points = df[dbscan.labels_ == label]
    print(len(cluster_points))
    centroid = cluster_points.mean(axis=0)
    codebook.append(centroid)
# print(len(codebook))
#%
np.save('/shared/racoon_fast/sim/codebook_init/dbscan_codebook.npy', codebook)
# torch.save(torch.from_numpy(kmeans.cluster_centers_), '/shared/racoon_fast/sim/codebook_init/codebook.pt')
#%
