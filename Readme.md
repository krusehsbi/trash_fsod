# Fewshot-Object Detection of Trash using synthetic data

## Setup

### Generate Fewshots

`python make_fewshot_yolo.py   --ann /pathtotaco/TACO/annotations.json   --images-root /pathtotaco/TACO/   --out /pathtotrashfsod/trash_fsod/datasets/3shot   --k 3   --classes "Plastic bag & wrapper,Cigarette,Bottle,Bottle cap,Can,Other plastic,Carton,Cup"  --val-ratio 0.3   --mode copy   --seed 42`

### (Option A) Append G-Shot

`python append_synth.py   --fewshot-dir /pathtotrashfsod/trash_fsod/datasets/3shot/   --synthetic-dir /pathtotrashfsod/trash_fsod/datasets/synthetic/   --out-dir /pathtotrashfsod/trash_fsod/datasets/k3_g30   --g 30   --seed 42`

### (Option B) Append G-Shot clustered

`python append_synth_cluster.py   --fewshot-dir /pathtotrashfsod/trash_fsod/datasets/1shot/   --synthetic-dir /pathtotrashfsod/trash_fsod/datasets/synthetic/   --out-dir /pathtotrashfsod/trash_fsod/datasets/k1_g10_cluster --g 10 --device cuda --model ViT-B/32
`

### Run training

`yolo detect train cfg=configs/k3_g30_train.yaml`

### Run test

`yolo detect val data=../../datasets/k3_g30/dataset.yaml split=test model=runs/k3_g30/yolov11_k3_g30/weights/best.pt`

## Results

### Base/finetune

yolov11 full training: 0.109/0.084
rt-detr full training: 0.23/0.172

**mAP50:**
|      | k=1  | k=3  | k=5  | k=10 | k=30 |
|------|------|------|------|------|------|
|Yolo  |0.108 |0.111 |0.138 |0.166 |0.213 |
|RTDETR|0.214 |0.219 |0.243 |0.292 |0.325 |

**mAP@[0.5-0.95]:**
|      | k=1  | k=3  | k=5  | k=10 | k=30 |
|------|------|------|------|------|------|
|Yolo  |0.0824|0.0826|0.101 |0.129 |0.161 |
|RTDETR|0.161 |0.163 |0.183 |0.225 |0.251 |


### YOLOv11

yolov11 full training: 0.109/0.084

**mAP50:**
|      | k=1  | k=3  | k=5  | k=10 | k=30 |
|------|------|------|------|------|------|
| g=0  |0.0517|0.0656|0.0735|0.129 |0.172 |
| g=10 |0.109 |0.116 |0.133 |0.165 |0.202 |
| g=20 |0.109 |0.115 |0.134 |0.162 |0.206 |
| g=30 |0.118 |0.125 |0.13  |0.163 |0.204 |


**mAP@[0.5-0.95]:**
|      | k=1  | k=3  | k=5  | k=10 | k=30 |
|------|------|------|------|------|------|
| g=0  |0.0428|0.0454|0.0558|0.0931|0.13  |
| g=10 |0.0835|0.0893|0.0989|0.128 |0.153 |
| g=20 |0.0819|0.0873|0.0998|0.125 |0.153 |
| g=30 |0.089 |0.0896|0.0968|0.124 |0.152 |

### YOLOv11 (clustered)

**mAP50:**
|      | k=1  | k=3  | k=5  | k=10 | k=30 |
|------|------|------|------|------|------|
| g=0  |0.0517|0.0656|0.0735|0.129 |0.172 |
| g=10 |0.12  |0.133 |0.143 |0.164 |0.204 |
| g=20 |0.107 |0.122 |0.141 |0.16  |0.216 |
| g=30 |0.111 |0.109 |0.137 |0.173 |0.212 |


**mAP@[0.5-0.95]:**
|      | k=1  | k=3  | k=5  | k=10 | k=30 |
|------|------|------|------|------|------|
| g=0  |0.0428|0.0454|0.0558|0.0931|0.13  |
| g=10 |0.0897|0.0965|0.105 |0.124 |0.153 |
| g=20 |0.0805|0.0944|0.105 |0.125 |0.164 |
| g=30 |0.0817|0.0827|0.104 |0.13  |0.16  |

________________________

### RT-DETR (clustered)

rt-detr full training: 0.23/0.172

**mAP50:**
|      | k=1   | k=3   | k=5  | k=10 | k=30 |
|------|-------|-------|------|------|------|
| g=0  |0.00355|0.00783|0.121 |0.111 |0.212 |
| g=10 |0.182  |0.163  |0.173 |0.193 |0.269 |
| g=20 |0.165  |0.161  |0.169 |0.22  |0.289 |
| g=30 |0.164  |0.181  |0.181 |0.201 |0.264 |

**mAP@[0.5-0.95]:**
|      | k=1   | k=3   | k=5  | k=10 | k=30 |
|------|-------|-------|------|------|------|
| g=0  |0.00215|0.00564|0.094 |0.0826|0.163 |
| g=10 |0.138  |0.125  |0.13  |0.153 |0.214 |
| g=20 |0.126  |0.121  |0.126 |0.172 |0.223 |
| g=30 |0.125  |0.135  |0.136 |0.156 |0.212 |

______________________________________________

### RT-DETR (SDXL base training)

Base: 0.174/0.122

**mAP50:**
| k=1   | k=3  | k=5  | k=10 | k=30 |
|-------|------|------|------|------|
|0.131  |0.155 |0.162 |0.207 |0.214 |

**mAP@[0.5-0.95]:**
| k=1   | k=3  | k=5  | k=10 | k=30 |
|-------|------|------|------|------|
|0.0609 |0.11  |0.117 |0.153 |0.161 |
