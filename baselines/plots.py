import matplotlib.pyplot as plt

# Data
k_values = [1, 5, 10, 30]

yolo = [0.108, 0.138, 0.166, 0.213]
rtdetr = [0.142, 0.143, 0.202, 0.283]

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
