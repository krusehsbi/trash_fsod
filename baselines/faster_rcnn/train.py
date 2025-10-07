#!/usr/bin/env python3
"""
Faster R-CNN finetuning on a COCO-style dataset.
- Loads pretrained weights (COCO) from torchvision.
- Trains on TRAIN, validates on VAL each epoch; saves best.pth by val mAP
- After training, reloads best.pth and evaluates on TEST

"""

from pathlib import Path
import json, random
import torch, torchvision, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from pycocotools.coco import COCO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from PIL import Image
import argparse

# ============================================================
# ---------------------- CONFIG LOADING -----------------------
# ============================================================

def load_config(cfg_path: str):
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


parser = argparse.ArgumentParser(description="Faster R-CNN base training")
parser.add_argument("--config", required=True, help="Path to config.json")
args = parser.parse_args()

cfg = load_config(args.config)

DATA_ROOT = Path(cfg["data_root"])
TRAIN_IMG_DIR = DATA_ROOT / "images/train"
VAL_IMG_DIR   = DATA_ROOT / "images/val"
TEST_IMG_DIR  = DATA_ROOT / "images/test"

TRAIN_JSON = DATA_ROOT / "annotations/instances_train.json"
VAL_JSON   = DATA_ROOT / "annotations/instances_val.json"
TEST_JSON  = DATA_ROOT / "annotations/instances_test.json"

OUT_DIR = Path(cfg["out_dir"])
SEED = cfg["seed"]
EPOCHS = cfg["epochs"]
BATCH_SIZE = cfg["batch_size"]
LR = cfg["lr"]
MOMENTUM = cfg["momentum"]
WEIGHT_DECAY = cfg["weight_decay"]
NUM_WORKERS = cfg["num_workers"]
MAX_GRAD_NORM = cfg["max_grad_norm"]
EVAL_INTERVAL = cfg["eval_interval"]
AMP = cfg["amp"]
FREEZE_BACKBONE = cfg["freeze_backbone"]
TRAINABLE_BACKBONE_LAYERS = cfg["trainable_backbone_layers"]
USE_COSINE_LR = cfg["use_cosine_lr"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# -------------------- DATASET CLASS -------------------------
# ============================================================

class CocoDetectionTorchvision(Dataset):
    """COCO -> (image_tensor, target_dict) for torchvision detection models."""
    def __init__(self, img_dir: Path, ann_file: Path, train: bool = True):
        self.coco = COCO(str(ann_file))
        self.img_dir = Path(img_dir)
        self.ids = list(sorted(self.coco.getImgIds()))
        if train:
            self.ids = [i for i in self.ids if len(self.coco.getAnnIds(imgIds=i, iscrowd=None)) > 0]

        cats = self.coco.loadCats(self.coco.getCatIds())
        # Map possibly non-contiguous COCO IDs to contiguous labels 1..K (0 is background)
        self.catid2contig = {int(c["id"]): i + 1 for i, c in enumerate(sorted(cats, key=lambda x: int(x["id"])))} 
        self.num_classes = len(self.catid2contig) + 1

        aug = [T.ToTensor()]
        if train:
            aug.append(T.RandomHorizontalFlip(0.5))
        self.transforms = T.Compose(aug)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=None))

        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            if int(a.get("iscrowd", 0)) == 1:
                continue
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])     # xyxy
            labels.append(self.catid2contig[int(a["category_id"])])
            areas.append(w * h)
            iscrowd.append(0)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas  = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        return self.transforms(img), target


def collate_fn(batch):
    return tuple(zip(*batch))

# ============================================================
# ---------------------- MODEL -------------------------------
# ============================================================

def create_model(num_classes: int):
    """
    Load a COCO-pretrained Faster R-CNN and
    replace the prediction head with the right number of classes.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",                           # pretrained COCO weights
        trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS,
    )
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False
    return model

# ============================================================
# -------------------- TRAINING UTILS ------------------------
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_ckpt(path: Path, model, optimizer, epoch: int, best_map: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_map": best_map,
    }, str(path))

@torch.no_grad()
def evaluate_map(model, loader, device, desc="Eval"):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    for images, targets in tqdm(loader, desc=desc):
        images = [img.to(device) for img in images]
        outputs = model(images)

        preds, gts = [], []
        for out, tgt in zip(outputs, targets):
            preds.append({
                "boxes": out["boxes"].cpu(),
                "scores": out["scores"].cpu(),
                "labels": out["labels"].cpu(),
            })
            gts.append({
                "boxes": tgt["boxes"].cpu(),
                "labels": tgt["labels"].cpu(),
            })
        metric.update(preds, gts)

    res = metric.compute()  # dict of tensors

    # Scalars TorchMetrics usually provides
    scalar_keys = [
        "map", "map_50", "map_75",
        "map_small", "map_medium", "map_large",
        "mar_1", "mar_10", "mar_100",
        "mar_small", "mar_medium", "mar_large",
    ]

    out = {}
    for k in scalar_keys:
        if k in res and res[k].ndim == 0:   # scalar tensor
            out[k] = float(res[k].item())

    # Optional per-class tensors -> lists (safe for JSON/logging)
    if "classes" in res:
        out["classes"] = res["classes"].cpu().tolist()
    if "precision" in res:
        out["precision_per_class"] = res["precision"].cpu().tolist()
    if "recall" in res:
        out["recall_per_class"] = res["recall"].cpu().tolist()

    # Return the main metric plus the full, JSON-safe dict
    main_map = float(res.get("map_50", torch.tensor(0.0)).item()) if "map_50" in res else 0.0
    return main_map, out

def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    avg_loss = 0.0
    for images, targets in tqdm(loader, desc="Train"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
        else:
            with torch.cuda.amp.autocast(enabled=AMP):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

        avg_loss = 0.9 * avg_loss + 0.1 * float(loss.item()) if avg_loss > 0 else float(loss.item())
    return avg_loss

# ============================================================
# ---------------------- MAIN --------------------------------
# ============================================================

def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Datasets & loaders
    train_ds = CocoDetectionTorchvision(TRAIN_IMG_DIR, TRAIN_JSON, train=True)
    val_ds   = CocoDetectionTorchvision(VAL_IMG_DIR,   VAL_JSON,   train=False)
    test_ds  = CocoDetectionTorchvision(TEST_IMG_DIR,  TEST_JSON,  train=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    # Model (pretrained) + optimizer + scheduler
    model = create_model(train_ds.num_classes).to(DEVICE)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    if USE_COSINE_LR:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    else:
        milestones = [int(EPOCHS * 0.6), int(EPOCHS * 0.85)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    # Training loop with val mAP tracking
    best_map, best_epoch = -1.0, -1
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_dl, optimizer, DEVICE, scaler)
        scheduler.step()
        print(f"Avg Train Loss: {train_loss:.4f}")

        if (epoch + 1) % EVAL_INTERVAL == 0:
            val_map, _ = evaluate_map(model, val_dl, DEVICE, desc="Eval (val)")
            print(f"Val mAP: {val_map:.4f}")
            save_ckpt(OUT_DIR / "last.pth", model, optimizer, epoch, best_map)
            if val_map > best_map:
                best_map, best_epoch = val_map, epoch
                save_ckpt(OUT_DIR / "best.pth", model, optimizer, epoch, best_map)
                print("-> Best model updated.")

    print(f"Training done. Best val mAP: {best_map:.4f} at epoch {best_epoch}.")

    # Final test evaluation with best checkpoint
    print("\n[Final] Evaluating BEST checkpoint on TEST set…")
    ckpt = torch.load(str(OUT_DIR / "best.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    test_map, test_detail = evaluate_map(model, test_dl, DEVICE, desc="Eval (test)")
    print(f"TEST mAP: {test_map:.4f}")

    # Save metrics
    (OUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "metrics/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test_map": test_map, **test_detail}, f, indent=2)
    print("Saved:", OUT_DIR / "metrics/test_metrics.json")


if __name__ == "__main__":
    main()
