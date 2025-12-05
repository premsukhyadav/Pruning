import torch
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math

import torch
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
from sklearn.manifold import MDS
import networkx as nx

def compute_cosine_similarity(conv_weights, conv_name, threshold=0.8,
                              consider_negative=True, plot=False):

    # number of filters in this convolution layer
    out_channels = conv_weights.shape[0]

    # reshape each filter to a single vector for cosine similarity
    flat = conv_weights.view(out_channels, -1)

    # normalize each filter vector so dot products become cosine similarity
    flat = F.normalize(flat, p=2, dim=1)

    # compute cosine similarity matrix (filter i · filter j)
    sim_matrix = torch.mm(flat, flat.t())

    # detect redundant filters
    redundant_pairs = []
    for i in range(out_channels):
        for j in range(i + 1, out_channels):
            sim = sim_matrix[i, j].item()
            if (sim > threshold) or (consider_negative and sim < -threshold):
                redundant_pairs.append((i, j, sim))

    # compute norms to decide which filter is weaker
    norms = torch.norm(conv_weights.view(out_channels, -1), dim=1)

    # choose which filters to drop
    drop_list = []
    for i, j, sim in redundant_pairs:
        drop = j if norms[j] < norms[i] else i
        drop_list.append(drop)

    # remove duplicates in drop list
    drop_list = sorted(set(drop_list))

    if plot:
        base_path = f"temp/{conv_name}/cosine_similarity"
        os.makedirs(base_path, exist_ok=True)

        # -------------------------------
        # 1) Standard heatmap
        # -------------------------------
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix.cpu().numpy(), cmap="coolwarm")
        plt.title(f"Cosine Similarity Heatmap ({conv_name})")
        plt.savefig(f"{base_path}/cosine_sim.png", bbox_inches='tight')


    return sim_matrix, redundant_pairs, drop_list




def analyze_multivariate(conv_weights, conv_name, lambda_std=1.0, meanstd_threshold_factor=0.5, plot=True):
    base_path = f"temp/{conv_name}"
    os.makedirs(base_path, exist_ok=True)

    # ---- Cosine Similarity ----
    sim_matrix, redundant_pairs, sim_drop = compute_cosine_similarity(conv_weights, conv_name, threshold=0.8, plot=True)
    print(f"\t\t-> Cosine Sim drop list: {sim_drop}")
    df_sim = pd.DataFrame(sim_matrix.cpu().numpy())
    df_sim.to_csv(f"{base_path}/cosine_similarity/cosine_similarity.csv", index=False)

    # --------------------------------------------------------
    # GLOBAL DROP LIST (Union of all drop methods)
    # --------------------------------------------------------
    combined_drop = sorted(set(sim_drop))
    return combined_drop
