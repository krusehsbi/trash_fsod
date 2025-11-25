import matplotlib.pyplot as plt

# Common x-axis (k values)
k = [1, 3, 5, 10, 30]

# ========================
# YOLOv11 (clustered)
# ========================

map50_clustered = {
    0: [0.00355, 0.00783, 0.121, 0.111, 0.212],
    10: [0.182, 0.163, 0.173, 0.193, 0.269],
    20: [0.165, 0.161, 0.169, 0.22, 0.289],
    30: [0.164, 0.181, 0.181, 0.201, 0.264]
}

map5095_clustered = {
    0:  [0.00215, 0.00564, 0.094, 0.0826, 0.163],
    10: [0.138,   0.125,   0.13,  0.153,  0.214],
    20: [0.126,   0.121,   0.126, 0.172,  0.223],
    30: [0.125,   0.135,   0.136, 0.156,  0.212],
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

# Clustered
plot_map(map50_clustered, "RT-DETR (clustered) for different k,g values — mAP50", "mAP50")
plot_map(map5095_clustered, "RT-DETR (clustered) for different k,g values — mAP@[0.5–0.95]", "mAP@[0.5–0.95]")

