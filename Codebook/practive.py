#%

#%
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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
data = total.numpy()


#%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Assuming `data` is your dataset as a numpy array or pandas DataFrame
wcss = []
silhouette_scores = []
K = range(256, 2049, 512)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=2)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data, kmeans.labels_))

# Elbow Method
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(K, wcss, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')

# Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')

plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# # Perform k-means clustering
# kmeans = KMeans(n_clusters=1024, random_state=42)
# labels = kmeans.fit_predict(data)
labels = kmeans.labels_
# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data)

# Plot t-SNE results with k-means labels
plt.figure(figsize=(10, 7))
scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter, ticks=range(4))
plt.title('t-SNE Visualization of k-means Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()