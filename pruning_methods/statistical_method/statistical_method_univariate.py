import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math


def compute_conv_l1_importance(conv_weights, conv_name, plot=False):
    base_path = f"temp/{conv_name}/l1_importance/"

    # Compute L1 norm for each filter (sum of absolute values of all weights in the filter)
    importance = torch.sum(torch.abs(conv_weights), dim=(1, 2, 3))

    # ------------------------------
    # Automatic drop based on magnitude
    # ------------------------------
    # Drop filters whose L1 norm < 0.25 (very small magnitude filters)
    drop_indices = [i for i, val in enumerate(importance) if val < 0.25]

    # ------------------------------
    # Optional plotting of L1 norms
    # ------------------------------
    if plot:
        os.makedirs(base_path, exist_ok=True)  # create folder if not exists
        plt.figure(figsize=(10, 5))
        x = range(len(importance))
        plt.bar(x, importance.cpu().numpy(), color='skyblue', label='Filter Importance')
        # draw horizontal line at 0.25 threshold
        plt.axhline(0.25, color='red', linestyle='--', label='Drop Threshold = 0.25')
        plt.title(f"L1 Importance per Filter: {conv_name}")
        plt.xlabel("Filter Index")
        plt.ylabel("L1 Norm (Importance)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{base_path}/l1_norm.png", bbox_inches='tight')
        plt.close()

    return importance, drop_indices


def compute_singular_values(conv_weights, conv_name, plot=False):
    # create folder for saving plots
    base_path = f"temp/{conv_name}/singular_values"
    os.makedirs(base_path, exist_ok=True)

    out_channels, in_channels, kH, kW = conv_weights.shape
    all_singular_values = []

    # compute singular values for each filter
    for i in range(out_channels):
        # flatten filter to 2D [in_channels, kH*kW] for SVD
        W = conv_weights[i].view(in_channels, -1)
        s = torch.linalg.svdvals(W)  # compute all singular values
        all_singular_values.append(s.cpu().numpy())  # convert to numpy and store

    all_singular_values = np.stack(all_singular_values)  # convert list to array [num_filters, num_singulars]

    # compute max and min singular value for each filter
    sigma_max = all_singular_values[:, 0]   # largest singular value
    sigma_min = all_singular_values[:, -1]  # smallest singular value

    # importance = largest + (spread between largest and smallest)
    importance = sigma_max + (sigma_max - sigma_min)

    # normalize importance to [0,1] for layer-consistent thresholding
    importance_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8) # To avoid divide by zero


    prune_threshold = 0.03 # TODO make this dynamic


    # suggest filters to drop based on normalized importance threshold
    drop_indices = np.where(importance_norm < prune_threshold)[0].tolist()

    # plot singular value spread per filter as horizontal line
    if plot:
        plt.figure(figsize=(60, 35))
        for i in range(out_channels):
            s_vals = all_singular_values[i]
            y = np.full_like(s_vals, i)  # y-coordinate = filter index
            plt.plot(s_vals, y, 'o-', alpha=0.65)  # plot points connected by line

        plt.title(f"{conv_name} — Singular Value Spread (Per Filter)")
        plt.xlabel("Singular Value Magnitude")
        plt.ylabel("Filter Index")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"{base_path}/singular_value_distance_plot.png", bbox_inches='tight')
        plt.close()

    # return all singular values, importance scores, and drop indices
    return all_singular_values, importance_norm, drop_indices


def analyze_univariate(conv_weights, conv_name, lambda_std=1.0, meanstd_threshold_factor=0.5, plot=False):
    base_path = f"temp/{conv_name}"
    os.makedirs(base_path, exist_ok=True)

    # ---- L1 Importance ----
    l1_importance, l1_drop = compute_conv_l1_importance(conv_weights, conv_name, plot=False)
    print(f"\t\t-> L1 drop list: {l1_drop}")
    df_l1 = pd.DataFrame({"Filter": list(range(len(l1_importance))),
                          "L1_Importance": l1_importance.cpu().numpy(),
                          "Drop_Suggested": [i in l1_drop for i in range(len(l1_importance))]})
    # Save to CSV
    csv_path = f"{base_path}/l1_importance/l1_importance.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_l1.to_csv(csv_path, index=False)


    # ---- Spectral Norm ----
    all_singular_values, importance_norm, sv_drop = compute_singular_values(conv_weights, conv_name, plot=plot)
    print(f"\t\t-> Singular value drop list: {sv_drop}")

    num_filters, num_sv = all_singular_values.shape

    # Build dataframe: singular values + importance + drop flag
    df_spec = pd.DataFrame(
        all_singular_values,  # singular values as columns
        columns=[f"S{i + 1}" for i in range(num_sv)]
    )
    df_spec.insert(0, "Filter", np.arange(num_filters))  # add filter index
    df_spec["Importance"] = importance_norm  # add normalized importance
    df_spec["Drop_Suggested"] = [i in sv_drop for i in range(num_filters)]  # mark drops

    # Save to CSV
    csv_path = f"{base_path}/singular_values/singular_values.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_spec.to_csv(csv_path, index=False)

    # --------------------------------------------------------
    # GLOBAL DROP LIST (Union of all drop methods)
    # --------------------------------------------------------
    combined_drop = sorted(set(l1_drop + sv_drop))
    return combined_drop
