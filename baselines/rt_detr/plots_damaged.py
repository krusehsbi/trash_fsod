import matplotlib.pyplot as plt

# Data
k_values = [1, 3, 5, 10, 30]

rtdetr = [0.214, 0.219, 0.243, 0.292, 0.325]
rtdetr_damaged = [0.214, 0.216, 0.245, 0.299, 0.324]

# Create evenly spaced positions for x-axis
x = range(len(k_values))   # [0,1,2,3]

plt.figure(figsize=(8, 5))

plt.plot(x, rtdetr_damaged, marker='x', linestyle='--', label="RT-DETR", color='grey')
plt.plot(x, rtdetr, marker='x', linestyle='--',label="RT-DETR (Deformed Objects)")

# Replace x-axis numbers with actual k labels
plt.xticks(x, k_values)

plt.title("Comparison of mAP50 for RT-DETR with and without deformed objects finetuned on k real objects.")
plt.xlabel("k")
plt.ylabel("mAP50")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()