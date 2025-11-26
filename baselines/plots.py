import matplotlib.pyplot as plt

# Data
k_values = [1, 3, 5, 10, 30]

yolo = [0.108, 0.111, 0.138, 0.166, 0.213]
rtdetr = [0.142, 0.138, 0.143, 0.202, 0.283]
rtdetr_sdxl = [0.131, 0.155, 0.162, 0.207, 0.214]

# Create evenly spaced positions for x-axis
x = range(len(k_values))   # [0,1,2,3]

plt.figure(figsize=(8, 5))

plt.plot(x, yolo, marker='x', linestyle='--', label="YOLO")
plt.plot(x, rtdetr, marker='x', linestyle='--',label="RT-DETR")

# Replace x-axis numbers with actual k labels
plt.xticks(x, k_values)

plt.title("Comparison of mAP50 for RT-DETR and YOLOv11 finetuned on k real objects.")
plt.xlabel("k")
plt.ylabel("mAP50")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(x, rtdetr, marker='x', linestyle='--', label="without SDXL")
plt.plot(x, rtdetr_sdxl, marker='x', linestyle='--',label="with SDXL")

# Replace x-axis numbers with actual k labels
plt.xticks(x, k_values)

plt.title("Comparison of mAP50 for RT-DETR with and without SDXL augmentations.")
plt.xlabel("k")
plt.ylabel("mAP50")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
