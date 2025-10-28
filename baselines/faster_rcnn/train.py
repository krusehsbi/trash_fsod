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
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

# ==============================
# CONFIG — edit these variables
# ==============================
EPOCHS: int = 100
BATCH: int = 4
IMG_SIZE: int = 1024
WORKERS: int = 4
OPTIMIZER: str = "SGD"
LR0: float = 0.0001
MOMENTUM: float = 0.937
WEIGHT_DECAY: float = 5e-4
LRF: float = 0.05
FREEZE: int = 0
PRETRAINED: bool = True
USE_AMP: bool = True
COLOR_JITTER: bool = True
SEED: int = 42
# Early stopping
EARLY_STOPPING_PATIENCE: int = 20
EARLY_STOPPING_MIN_DELTA: float = 1e-4
# ==============================
# --- Augmentation knobs ---
STRONG_AUG: bool = True          # master switch
SCALE_JITTER: tuple = (0.8, 1.2) # uniform scale factor range
GRAYSCALE_P: float = 0.1
AUTOCONTRAST_P: float = 0.2
SHARPNESS_P: float = 0.2
BLUR_P: float = 0.15              # Gaussian blur probability
CUTOUT_P: float = 0.5             # do N cutout holes with this prob
CUTOUT_HOLES: int = 8
CUTOUT_RATIO: tuple = (0.02, 0.10)  # area ratio per hole w.r.t image
GAUSS_NOISE_P: float = 0.3
GAUSS_NOISE_STD: float = 0.02     # relative to [0,1] range


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

        # existing light augs
        self.flip = transforms.RandomHorizontalFlip(0.5) if train else None
        self.color = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02) if (train and enable_color_jitter) else None

        # stateless PIL ops we'll call conditionally
        self.autocontrast = transforms.functional.autocontrast
        self.adjust_sharpness = transforms.functional.adjust_sharpness
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        self.to_tensor = F.to_tensor

    def __len__(self):
        return len(self.images)

    def _load_targets(self, img_path: Path, W: int, H: int):
        # (unchanged from your version)
        try:
            rel = img_path.relative_to(self.images_dir)
            label_path = (self.labels_dir / rel).with_suffix('.txt')
        except Exception:
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
        labels = []
        if raw_labels:
            max_lbl = max(raw_labels)
            if max_lbl >= len(self.class_names):
                raw_labels = [r - 1 for r in raw_labels]
            labels = [r + 1 for r in raw_labels]  # torchvision expects 1..C (0 is background)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        return {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd}

    # ---------- helpers for strong aug ----------
    def _scale_jitter(self, image: Image.Image, target, s_min=0.8, s_max=1.2):
        W, H = image.size
        s = random.uniform(s_min, s_max)
        if abs(s - 1.0) < 1e-3:
            return image, target
        newW, newH = max(1, int(W * s)), max(1, int(H * s))
        image = image.resize((newW, newH), Image.BILINEAR)
        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= s
            boxes[:, [1, 3]] *= s
            # clip to new image size
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, newW)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, newH)
            target["boxes"] = boxes
            # recompute area
            target["area"] = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        return image, target

    def _maybe_gray(self, image: Image.Image):
        if random.random() < GRAYSCALE_P:
            return transforms.functional.to_grayscale(image, num_output_channels=3)
        return image

    def _maybe_autocontrast(self, image: Image.Image):
        if random.random() < AUTOCONTRAST_P:
            return self.autocontrast(image)
        return image

    def _maybe_sharpness(self, image: Image.Image):
        if random.random() < SHARPNESS_P:
            factor = random.uniform(0.5, 2.0)
            return self.adjust_sharpness(image, factor)
        return image

    def _maybe_blur(self, image: Image.Image):
        if random.random() < BLUR_P:
            return self.gaussian_blur(image)
        return image

    def _apply_cutout(self, image: torch.Tensor):
        # image is tensor in [0,1], shape [C,H,W]
        if random.random() >= CUTOUT_P:
            return image
        C, H, W = image.shape
        holes = CUTOUT_HOLES
        min_r, max_r = CUTOUT_RATIO
        for _ in range(holes):
            area = random.uniform(min_r, max_r) * H * W
            aspect = random.uniform(0.5, 2.0)
            h = int((area / aspect) ** 0.5)
            w = int(area / max(h, 1))
            if h <= 0 or w <= 0:
                continue
            y = random.randint(0, max(H - h, 0))
            x = random.randint(0, max(W - w, 0))
            image[:, y:y+h, x:x+w] = 0.0  # black rectangle
        return image
    # -------------------------------------------

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        W, H = image.size
        target = self._load_targets(img_path, W, H)
        target["image_id"] = torch.tensor([idx])

        if not self.train:
            return F.to_tensor(image), target

        # --- STRONG AUG chain (PIL domain, bbox-safe) ---
        if STRONG_AUG:
            # photometric
            image = self._maybe_autocontrast(image)
            image = self._maybe_sharpness(image)
            if self.enable_color_jitter:  # respect your toggle
                image = self.color(image)  # existing jitter
            image = self._maybe_gray(image)
            image = self._maybe_blur(image)

            # geometric: scale jitter with bbox rescale
            image, target = self._scale_jitter(
                image, target, s_min=SCALE_JITTER[0], s_max=SCALE_JITTER[1]
            )
            target = self._clip_and_filter(target, *image.size)  # (W,H)

        # existing H-flip (bbox-correct)
        if self.flip and random.random() < 0.5:
            image = F.hflip(image)
            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                newW = image.size[0]
                boxes[:, [0, 2]] = newW - boxes[:, [2, 0]]
                target["boxes"] = boxes
            target = self._clip_and_filter(target, *image.size)  # (W,H)

        # to tensor
        image = self.to_tensor(image)

        # tensor-domain augs
        if STRONG_AUG:
            # light Gaussian noise
            if random.random() < GAUSS_NOISE_P:
                noise = torch.randn_like(image) * GAUSS_NOISE_STD
                image = (image + noise).clamp(0.0, 1.0)
            # cutout occlusion
            image = self._apply_cutout(image)

        return image, target

    def _clip_and_filter(self, target, W: int, H: int, eps: float = 1e-3):
        if target["boxes"].numel() == 0:
            return target
        boxes = target["boxes"]

        # clip to (almost) inside image to avoid 0-size after clamp
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W - eps)  # x1,x2
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H - eps)  # y1,y2

        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        keep = (w > 0) & (h > 0)

        target["boxes"]  = boxes[keep]
        target["labels"] = target["labels"][keep]
        target["iscrowd"] = target["iscrowd"][keep]
        if "area" in target:
            target["area"] = (target["boxes"][:, 2] - target["boxes"][:, 0]).clamp(min=0) * \
                            (target["boxes"][:, 3] - target["boxes"][:, 1]).clamp(min=0)
        return target


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
    # Build model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)

    # Replace the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    # Better anchors. add smaller anchors + more ratios
    # Per FPN level (P3..P7 effectively).
    sizes = ((16,), (32,), (64,), (128,), (256,))
    ratios = (0.5, 1.0, 2.0, 3.0)
    ag = AnchorGenerator(sizes=sizes, aspect_ratios=(ratios,)*5)
    model.rpn.anchor_generator = ag

    # Rebuild RPN head to match new num_anchors_per_location
    out_channels = model.backbone.out_channels  # usually 256
    num_anchors = ag.num_anchors_per_location()[0]  # e.g., 4 ratios => 4
    model.rpn.head = RPNHead(out_channels, num_anchors)

    # RPN sampling / thresholds (more recall on tiny data)
    model.rpn.batch_size_per_image = 512
    model.rpn.positive_fraction   = 0.5       # 0.5 helps few-shot
    model.rpn.score_thresh        = 0.0
    model.rpn.nms_thresh          = 0.7
    model.rpn.pre_nms_top_n_train  = 4000     # more proposals while training
    model.rpn.pre_nms_top_n_test   = 2000
    model.rpn.post_nms_top_n_train = 2000
    model.rpn.post_nms_top_n_test  = 1000
    model.rpn_fg_iou_thresh       = 0.5
    model.rpn_bg_iou_thresh       = 0.3

    # ROI head sampler (get more positives to learn faster)
    model.roi_heads.batch_size_per_image = 512
    model.roi_heads.positive_fraction    = 0.5   # default 0.25 increase to learn faster in low-data
    model.roi_heads.nms_thresh           = 0.5
    model.roi_heads.score_thresh         = 0.0
    model.roi_heads.detections_per_img   = 300

    #Progressive freezing
    if freeze and hasattr(model.backbone, 'body'):
        freeze_map = {
            1: ('conv1','bn1','layer1'),
            2: ('conv1','bn1','layer1','layer2'),
            3: ('conv1','bn1','layer1','layer2','layer3'),
            4: ('conv1','bn1','layer1','layer2','layer3','layer4'),
        }
        to_freeze = freeze_map.get(freeze, ())
        for name, p in model.backbone.body.named_parameters():
            if any(name.startswith(pref) for pref in to_freeze):
                p.requires_grad = False

    # initialize cls bias ~ log(p/(1-p)) with small foreground prior p
    with torch.no_grad():
        cls_score = model.roi_heads.box_predictor.cls_score
        p = 0.01
        bias = torch.full_like(cls_score.bias, fill_value=torch.log(torch.tensor(p/(1-p))))
        cls_score.bias.copy_(bias)

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

    def is_head(n): return n.startswith('roi_heads.box_predictor')
    def is_rpn(n):  return n.startswith('rpn')
    def is_fpn(n):  return n.startswith('backbone.fpn')

    lr_backbone = 5e-4   # keep small
    lr_heads    = 5e-3   # much larger so it actually learns

    backbone_params, fpn_rpn_params, head_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if is_head(n): head_params.append(p)
        elif is_rpn(n) or is_fpn(n): fpn_rpn_params.append(p)
        else: backbone_params.append(p)

    optimizer = optim.SGD([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": fpn_rpn_params,  "lr": lr_backbone},
        {"params": head_params,     "lr": lr_heads},
    ], momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=min(lr_backbone, lr_heads) * LRF
    )

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
