#!/usr/bin/env python3
"""
Train Faster R-CNN on a YOLO-format dataset (YOLO folder or data YAML).

Usage
-----
python train_frcnn_from_yolo.py \
    --data /path/to/yolo_root_or_dataset.yaml \
    --out runs/frcnn-5shot

Edit the hyperparameters in the CONFIG section below as desired.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import yaml
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# ==============================
# CONFIG — edit these variables
# ==============================
EPOCHS: int = 100
BATCH: int = 4
IMG_SIZE: int = 1024
WORKERS: int = 4
OPTIMIZER: str = "SGD"
LR0: float = 0.001
MOMENTUM: float = 0.937
WEIGHT_DECAY: float = 5e-4
LRF: float = 0.05
FREEZE: int = 0
PRETRAINED: bool = True
USE_AMP: bool = True
COLOR_JITTER: bool = False
SEED: int = 42
# Early stopping
EARLY_STOPPING_PATIENCE: int = 20
EARLY_STOPPING_MIN_DELTA: float = 1e-4
# ==============================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def yolo_denorm_to_xyxy(xc, yc, w, h, W, H):
    x = (xc - w / 2.0) * W
    y = (yc - h / 2.0) * H
    x2 = (xc + w / 2.0) * W
    y2 = (yc + h / 2.0) * H
    return max(0, x), max(0, y), min(W, x2), min(H, y2)

class YoloDetectionDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, class_names: List[str], max_size=None, train=False, enable_color_jitter=False):
        self.images = sorted([p for p in images_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        self.images_dir = images_dir  # preserve root to compute relative paths for labels
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.max_size = max_size
        self.train = train
        self.enable_color_jitter = enable_color_jitter
        self.flip = transforms.RandomHorizontalFlip(0.5) if train else None
        self.color = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02) if (train and enable_color_jitter) else None

    def __len__(self):
        return len(self.images)

    def _load_targets(self, img_path: Path, W: int, H: int):
        # preserve subfolder structure: map images/<subdirs>/img.jpg -> labels/<subdirs>/img.txt
        try:
            rel = img_path.relative_to(self.images_dir)
            label_path = (self.labels_dir / rel).with_suffix('.txt')
        except Exception:
            # fallback: same-dir label file or labels root with filename
            label_path = self.labels_dir / img_path.with_suffix('.txt').name
        boxes, raw_labels = [], []
        if label_path.exists():
            for line in open(label_path):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
                boxes.append(yolo_denorm_to_xyxy(xc, yc, w, h, W, H))
                raw_labels.append(cls)
        # Normalize class indices: detect 1-based files and convert to 0-based
        labels = []
        if raw_labels:
            max_lbl = max(raw_labels)
            if max_lbl >= len(self.class_names):
                # assume 1-based -> convert all to 0-based
                raw_labels = [r - 1 for r in raw_labels]
            labels = [r + 1 for r in raw_labels]  # torchvision expects 1..C (0 is background)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        return {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd}

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        W, H = image.size
        target = self._load_targets(img_path, W, H)
        target["image_id"] = torch.tensor([idx])
        if self.train and self.color:
            image = self.color(image)
        if self.train and self.flip:
            if random.random() < 0.5:
                image = F.hflip(image)
                if target["boxes"].numel() > 0:
                    boxes = target["boxes"].clone()
                    boxes[:, [0, 2]] = image.size[0] - boxes[:, [2, 0]]
                    target["boxes"] = boxes
        image = F.to_tensor(image)
        return image, target

def parse_yolo_data_yaml(yaml_path: Path):
    data = yaml.safe_load(open(yaml_path))
    names = data['names'] if isinstance(data['names'], list) else [data['names'][k] for k in sorted(data['names'])]
    train_path = Path(data['train'])
    val_path = Path(data.get('val') or data.get('valid'))
    if not train_path.is_absolute():
        train_path = yaml_path.parent / train_path
    if not val_path.is_absolute():
        val_path = yaml_path.parent / val_path
    return names, train_path, val_path

def build_model(num_classes: int, pretrained=True, freeze=0):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    if freeze and hasattr(model.backbone, 'body'):
        for name, param in model.backbone.body.named_parameters():
            if name.startswith(('conv1', 'bn1', 'layer1')):
                param.requires_grad = False
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, scaler=None):
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_loss(model, data_loader, device):
    model.train()
    total = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            total += sum(loss for loss in losses.values()).item()
    return total / len(data_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to YOLO dataset root or YAML')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()

    set_seed(SEED)
    data_path = Path(args.data)

    if data_path.suffix in {'.yaml', '.yml'}:
        class_names, train_img_dir, val_img_dir = parse_yolo_data_yaml(data_path)
        def labels_for(img_dir):
            idx = list(img_dir.parts).index('images')
            return Path(*img_dir.parts[:idx]) / 'labels' / Path(*img_dir.parts[idx+1:])
        train_lbl, val_lbl = labels_for(train_img_dir), labels_for(val_img_dir)
    else:
        root = data_path
        class_names = [str(i) for i in range(1)]
        train_img_dir, train_lbl = root/'images/train', root/'labels/train'
        val_img_dir, val_lbl = root/'images/val', root/'labels/val'

    train_ds = YoloDetectionDataset(train_img_dir, train_lbl, class_names, IMG_SIZE, True, COLOR_JITTER)
    val_ds = YoloDetectionDataset(val_img_dir, val_lbl, class_names, IMG_SIZE, False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(len(class_names), PRETRAINED, FREEZE).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LR0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True) if OPTIMIZER.upper() == 'SGD' else optim.AdamW(params, lr=LR0, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR0 * LRF)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and device.type == 'cuda' else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')

    # early stopping state
    patience = EARLY_STOPPING_PATIENCE
    min_delta = EARLY_STOPPING_MIN_DELTA
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS+1):
        tr_loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        val_loss = evaluate_loss(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch}/{EPOCHS} - lr {optimizer.param_groups[0]['lr']:.6f} - train {tr_loss:.4f} - val {val_loss:.4f}")

        # check improvement
        if val_loss < best_val - min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / 'best.pt')
        else:
            epochs_no_improve += 1

        torch.save(model.state_dict(), out_dir / 'last.pt')

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered (no improvement for {patience} epochs). Stopping at epoch {epoch}.")
            break

    print(f"Training done. Best val loss: {best_val:.4f}")

if __name__ == '__main__':
    main()
