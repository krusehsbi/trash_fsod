import matplotlib.pyplot as plt

# Common x-axis (k values)
k = [1, 3, 5, 10, 30]

# ========================
# YOLOv11 (baseline)
# ========================

map50_base = {
    0: [0.0517, 0.0656, 0.0735, 0.129, 0.172],
    10: [0.109, 0.116, 0.133, 0.165, 0.202],
    20: [0.109, 0.115, 0.134, 0.162, 0.206],
    30: [0.118, 0.125, 0.13, 0.163, 0.204]
}

map5095_base = {
    0: [0.0428, 0.0454, 0.0558, 0.0931, 0.13],
    10: [0.0835, 0.0893, 0.0989, 0.128, 0.153],
    20: [0.0819, 0.0873, 0.0998, 0.125, 0.153],
    30: [0.089, 0.0896, 0.0968, 0.124, 0.152]
}

# ========================
# YOLOv11 (clustered)
# ========================

map50_clustered = {
    0: [0.0517, 0.0656, 0.0735, 0.129, 0.172],
    10: [0.12, 0.133, 0.143, 0.164, 0.204],
    20: [0.107, 0.122, 0.141, 0.16, 0.216],
    30: [0.111, 0.109, 0.137, 0.173, 0.212]
}

map5095_clustered = {
    0: [0.0428, 0.0454, 0.0558, 0.0931, 0.13],
    10: [0.0897, 0.0965, 0.105, 0.124, 0.153],
    20: [0.0805, 0.0944, 0.105, 0.125, 0.164],
    30: [0.0817, 0.0827, 0.104, 0.13, 0.16]
}

# ========================
# YOLOv11 (stable diffusion)
# ========================

map50_sd = {
    0: [0.0571, 0.0979, 0.133, 0.13, 0.174]
}

map5095_sd = {
    0: [0.0467, 0.0669, 0.0947, 0.0959, 0.131]
}

# ========================
# Plotting Function (with grey g=0)
# ========================

def plot_map(data, title, ylabel):
    plt.figure(figsize=(7, 5))
    x = range(len(k))  # evenly spaced x positions

    for g, values in data.items():
        if g == 0:
            color = 'grey'
            label = f"g={g}"
            plt.scatter(x, values, label=label, s=60, color=color, marker='x')
            plt.plot(x, values, linestyle='--', alpha=0.7, color=color)
        else:
            plt.scatter(x, values, label=f"g={g}", s=60, marker='x')
            plt.plot(x, values, linestyle='--', alpha=0.6)

    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel("k", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Group (g)")
    plt.xticks(x, k)
    plt.tight_layout()
    plt.show()


# ========================
# Generate All Plots
# ========================

# Baseline
plot_map(map50_base, "YOLOv11 for different k,g values — mAP50", "mAP50")
plot_map(map5095_base, "YOLOv11 for different k,g values — mAP@[0.5–0.95]", "mAP@[0.5–0.95]")

# Clustered
plot_map(map50_clustered, "YOLOv11 (clustered) for different k,g values — mAP50", "mAP50")
plot_map(map5095_clustered, "YOLOv11 (clustered) for different k,g values — mAP@[0.5–0.95]", "mAP@[0.5–0.95]")

# Stable Diffusion vs Clustered (g=0)
map50_sd_compare = {
    "Stable Diffusion (g=0)": map50_sd[0],
    "Clustered (g=0)": map50_clustered[0]
}
map5095_sd_compare = {
    "Stable Diffusion (g=0)": map5095_sd[0],
    "Clustered (g=0)": map5095_clustered[0]
}

# Reuse same plot function
plot_map(map50_sd_compare, "YOLOv11 (Stable Diffusion vs Clustered g=0) — mAP50", "mAP50")
plot_map(map5095_sd_compare, "YOLOv11 (Stable Diffusion vs Clustered g=0) — mAP@[0.5–0.95]", "mAP@[0.5–0.95]")

