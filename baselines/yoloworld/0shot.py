from ultralytics import YOLO
import yaml

# Create a YOLO-World model
model = YOLO("yolov8l-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes

with open('../../datasets/1shot/dataset.yaml', 'r') as f:
    data = yaml.safe_load(f)

# Extract class names
class_names = list(data['names'].values()) if isinstance(data['names'], dict) else data['names']

print("Classes:", class_names)

model.set_classes(class_names)

# Conduct model validation on the COCO8 example dataset
metrics = model.val(data="../../datasets/1shot/dataset.yaml", split="test", imgsz=1024,
                    batch=8, project="runs/yoloworld", name="0shot", save_json=True)