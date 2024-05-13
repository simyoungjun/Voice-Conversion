import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


def spk_emb_corr(x, quantized_, metric='cos'):
        # spk_emb fig plot
    # spk embedding correlation 확인
    spk = (x - quantized_).reshape(-1, 1024)

    # cosine metric
    if 'cos' in metric:
        spk_1 = spk.unsqueeze(0).expand(spk.size(0), -1, -1)
        spk_2 = spk.unsqueeze(1).expand(-1, spk.size(0), -1)
        cos_sim = F.cosine_similarity(spk_1, spk_2, dim=-1)
        heatmap = plt.imshow(cos_sim.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')
    else:
        # L2
        spk_1 = spk.unsqueeze(0)
        spk_2 = spk.unsqueeze(0)
        L2_dist=torch.cdist(spk_1, spk_2).squeeze()
        # Create the plot
        heatmap = plt.imshow(L2_dist.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')


    # Add a color bar
    plt.colorbar(heatmap)

    # Add title and labels as needed
    plt.title('2D Array Heat Map')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    # plt.savefig(f'eval_fig/spk_emb/{args.model_name}_cos.png')
    # plt.savefig(f'eval_fig/spk_emb/corr_{metric}.png')
    # Show the plot
    
    return plt