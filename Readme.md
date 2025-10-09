# Fewshot-Object Detection of Trash using synthetic data

## Setup

### Generate Fewshots

`python make_fewshot_yolo.py   --ann /pathtotaco/TACO/annotations.json   --images-root /pathtotaco/TACO/   --out /pathtotrashfsod/trash_fsod/datasets/3shot   --k 3   --classes "Plastic bag & wrapper,Cigarette,Bottle,Bottle cap,Can,Other plastic,Carton,Cup"  --val-ratio 0.3   --mode copy   --seed 42
`

### Append G-Shot

`python append_synth.py   --fewshot-dir /pathtotrashfsod/trash_fsod/datasets/3shot/   --synthetic-dir /pathtotrashfsod/trash_fsod/datasets/synthetic/   --out-dir /pathtotrashfsod/trash_fsod/datasets/k3_g30   --g 30   --seed 42
`

## Results

### YOLOv11

**mAP50:**
|      | k=1  | k=3  | k=5  | k=10 | k=30 |
|------|------|------|------|------|------|
| g=0  |0.0517|0.0656|0.0735|0.129 |0.172 |
| g=10 |      |0.116 |0.133 |0.165 |0.202 |
| g=20 |      |0.115 |0.134 |0.162 |0.206 |
| g=30 |      |0.125 |0.13  |0.163 |0.204 |


**mAP@[0.5-0.95]:**
|      | k=1  | k=3  | k=5  | k=10 | k=30 |
|------|------|------|------|------|------|
| g=0  |0.0428|0.0454|0.0558|0.0931|0.13  |
| g=10 |      |0.0893|0.0989|0.128 |0.153 |
| g=20 |      |0.0873|0.0998|0.125 |0.153 |
| g=30 |      |0.0896|0.0968|0.124 |0.152 |