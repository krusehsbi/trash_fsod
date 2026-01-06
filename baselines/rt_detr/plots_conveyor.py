import matplotlib.pyplot as plt

# Common x-axis (k values)
k = [1, 3, 5, 10, 30]

# use evenly spaced x positions so ticks are equally spaced on the axis
x_pos = list(range(len(k)))

map50 = [0.00922, 0.00926, 0.00946, 0.00984, 0.152]

map50_95 = [0.00461, 0.00465, 0.00472, 0.00507, 0.105]

map50_ft = [0.0242, 0.0237, 0.0231, 0.119, 0.21]

map50_95_ft = [0.013, 0.0126, 0.0121, 0.0788, 0.14]

# First plot: mAP@50
plt.figure(figsize=(7, 5))
plt.plot(x_pos, map50, color='grey', marker='x', linestyle='--', label='mAP@50')
plt.plot(x_pos, map50_ft, color='tab:blue', marker='x', linestyle='--', label='mAP@50 (fine-tuned)')
plt.title('mAP@50 vs k')
plt.xlabel('k')
plt.ylabel('mAP@50')
plt.xticks(x_pos, k)  # show original k values but place them evenly
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Second plot: mAP@50.95
plt.figure(figsize=(7, 5))
plt.plot(x_pos, map50_95, color='grey', marker='x', linestyle='--', label='mAP@50.95')
plt.plot(x_pos, map50_95_ft, color='tab:orange', marker='x', linestyle='--', label='mAP@50.95 (fine-tuned)')
plt.title('mAP@50.95 vs k')
plt.xlabel('k')
plt.ylabel('mAP@50.95')
plt.xticks(x_pos, k)  # show original k values but place them evenly
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()