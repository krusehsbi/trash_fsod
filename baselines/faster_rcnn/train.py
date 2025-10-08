#!/usr/bin/env python3
"""
Faster R-CNN finetuning on a COCO-style dataset, driven by a config file.

- Trains on TRAIN, validates on VAL each epoch; keeps best.pth by val mAP
- After training, reloads best.pth and evaluates on TEST
- Loads settings from --config (JSON or YAML)
- Optionally loads a base checkpoint (load_from) or resumes (resume_from)
- If class count differs when loading a base checkpoint, ROI head weights are ignored safely
"""

from pathlib import Path
import argparse, json, random
import torch, torchvision, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from pycocotools.coco import COCO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# ---------------------- Config loading ----------------------
def load_config(cfg_path: str) -> dict:
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if p.suffix.lower() in [".yml", ".yaml"]:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("PyYAML not installed. pip install pyyaml") from e
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        return json.loads(p.read_text(encoding="utf-8"))

parser = argparse.ArgumentParser(description="Faster R-CNN base training (config-driven)")
parser.add_argument("--config", required=True, help="Path to config .json/.yaml")
args = parser.parse_args()
CFG = load_config(args.config)

# ---------------------- Paths & hyperparams ----------------------
DATA_ROOT = Path(CFG["data_root"]).resolve()
TRAIN_IMG_DIR = DATA_ROOT / "images/train"
VAL_IMG_DIR   = DATA_ROOT / "images/val"
TEST_IMG_DIR  = DATA_ROOT / "images/test"

TRAIN_JSON = DATA_ROOT / "annotations/instances_train.json"
VAL_JSON   = DATA_ROOT / "annotations/instances_val.json"
TEST_JSON  = DATA_ROOT / "annotations/instances_test.json"

OUT_DIR = Path(CFG.get("out_dir", "runs")).resolve()
SEED = int(CFG.get("seed", 1337))
EPOCHS = int(CFG.get("epochs", 20))
BATCH_SIZE = int(CFG.get("batch_size", 8))
LR = float(CFG.get("lr", 0.02))
MOMENTUM = float(CFG.get("momentum", 0.9))
WEIGHT_DECAY = float(CFG.get("weight_decay", 1e-4))
NUM_WORKERS = int(CFG.get("num_workers", 4))
MAX_GRAD_NORM = float(CFG.get("max_grad_norm", 10.0))
EVAL_INTERVAL = int(CFG.get("eval_interval", 1))
AMP = bool(CFG.get("amp", False))
FREEZE_BACKBONE = bool(CFG.get("freeze_backbone", False))
TRAINABLE_BACKBONE_LAYERS = int(CFG.get("trainable_backbone_layers", 3))
USE_COSINE_LR = bool(CFG.get("use_cosine_lr", False))
EARLY_STOP_PATIENCE = int(CFG.get("early_stop_patience", 0))   # 0 disables
EARLY_STOP_MIN_DELTA = float(CFG.get("early_stop_min_delta", 0.0))

# Checkpoint behavior
LOAD_FROM = CFG.get("load_from", None)      # load weights only (good for finetune)
RESUME_FROM = CFG.get("resume_from", None)  # resume training (model + optimizer + epoch)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# -------------------- DATASET CLASS -------------------------
# ============================================================

class CocoDetectionTorchvision(Dataset):
    def __init__(self, img_dir: Path, ann_file: Path, train: bool = True):
        self.coco = COCO(str(ann_file))
        self.img_dir = Path(img_dir)
        self.ids = list(sorted(self.coco.getImgIds()))
        if train:
            self.ids = [i for i in self.ids if len(self.coco.getAnnIds(imgIds=i, iscrowd=None)) > 0]

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.catid2contig = {int(c["id"]): i + 1 for i, c in enumerate(sorted(cats, key=lambda x: int(x["id"])))} 
        self.num_classes = len(self.catid2contig) + 1

        self.train = train
        self.use_aug = bool(CFG.get("augmentation", False)) and train

        if self.use_aug:

            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(size=(1024, 1024), scale=(0.6, 1.0), ratio=(0.75, 1.33), p=0.3),
                A.ColorJitter(0.3, 0.3, 0.3, 0.1, p=0.8),
                A.HueSaturationValue(10, 20, 20, p=0.5),
                A.RGBShift(10, 10, 10, p=0.3),
                A.OneOf([A.Blur(3, p=1.0), A.GaussNoise(var_limit=(5.0, 20.0), p=1.0)], p=0.2),
                A.ToFloat(max_value=255.0),     # images -> float32 in [0,1]
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format="pascal_voc",            # [xmin, ymin, xmax, ymax] in PIXELS
                label_fields=["labels"],        # labels are provided separately
                min_visibility=0.2,
                check_each_transform=True       # stricter validation, useful for debugging
            ))
        else:
            self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(self.img_dir / path).convert("RGB")

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            xmin, ymin, w, h = ann["bbox"]
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(self.catid2contig[ann["category_id"]])
            areas.append(w * h)
            iscrowd.append(ann.get("iscrowd", 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        W, H = img.size  # PIL gives (W, H)

        if boxes.numel():
            # clamp into image bounds
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=W - 1e-3)  # x1,x2
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=H - 1e-3)  # y1,y2

            # drop boxes that lost validity
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, labels, areas, iscrowd = boxes[keep], labels[keep], areas[keep], iscrowd[keep]
        W, H = img.size  # PIL gives (W, H)

        if boxes.numel():
            # clamp into image bounds
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=W - 1e-3)  # x1,x2
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=H - 1e-3)  # y1,y2

            # drop boxes that lost validity
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, labels, areas, iscrowd = boxes[keep], labels[keep], areas[keep], iscrowd[keep]

        # initial target
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        # --- apply strong augmentation ---
        if self.use_aug:
            img_np = np.array(img)  # HWC, uint8
            bxs_list = boxes.tolist()
            lbs_list = labels.tolist()

            transformed = self.transform(
                image=img_np, 
                bboxes=bxs_list, 
                labels=lbs_list
            )

            img_t = transformed["image"]
            bxs_t = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            lbs_t = torch.tensor(transformed["labels"], dtype=torch.int64)

            if bxs_t.numel() == 0:
                bxs_t = torch.zeros((0, 4), dtype=torch.float32)
                lbs_t = torch.zeros((0,), dtype=torch.int64)

            target["boxes"] = bxs_t
            target["labels"] = lbs_t
            target["area"] = (
                (bxs_t[:, 2] - bxs_t[:, 0]).clamp(min=0) *
                (bxs_t[:, 3] - bxs_t[:, 1]).clamp(min=0)
            )
            img = img_t
        else:
            # no augmentation -> plain ToTensor()
            img = self.transform(img)

        return img, target



def collate_fn(batch):
    return tuple(zip(*batch))

# ============================================================
# ---------------------- MODEL -------------------------------
# ============================================================

def create_model(num_classes: int):
    """Load COCO-pretrained Faster R-CNN and swap the head."""
    backbone = resnet_fpn_backbone('resnet101', weights='DEFAULT', trainable_layers=5)
    model = FasterRCNN(backbone, num_classes=num_classes)

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

def load_ckpt_flex(model, ckpt_path: Path, load_optimizer=False, optimizer=None):
    """
    Load a checkpoint and ignore ROI head keys if shapes mismatch
    (useful for finetuning on different class counts).
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model", ckpt)  # support pure state_dict too

    # Drop ROI head keys — safest when num_classes changed
    drop_prefixes = [
        "roi_heads.box_predictor.cls_score",
        "roi_heads.box_predictor.bbox_pred",
    ]
    filtered = {}
    for k, v in state.items():
        if any(k.startswith(pref) for pref in drop_prefixes):
            continue
        # if target shape does not match, skip
        if k in model.state_dict() and model.state_dict()[k].shape != v.shape:
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[load_ckpt_flex] loaded: {len(filtered)} keys; missing={len(missing)} unexpected={len(unexpected)}")

    start_epoch = -1
    best_map = -1.0
    if load_optimizer and optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1))
        best_map = float(ckpt.get("best_map", -1.0))
    return start_epoch, best_map

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

    # Scalars
    scalar_keys = [
        "map", "map_50", "map_75",
        "map_small", "map_medium", "map_large",
        "mar_1", "mar_10", "mar_100",
        "mar_small", "mar_medium", "mar_large",
    ]
    out = {}
    for k in scalar_keys:
        if k in res and res[k].ndim == 0:
            out[k] = float(res[k].item())

    # Per-class arrays for saving/inspection
    if "classes" in res:
        out["classes"] = res["classes"].cpu().tolist()
    if "precision" in res:
        out["precision_per_class"] = res["precision"].cpu().tolist()
    if "recall" in res:
        out["recall_per_class"] = res["recall"].cpu().tolist()

    main_map = float(res.get("map", torch.tensor(0.0)).item()) if "map" in res else 0.0
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
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    if USE_COSINE_LR:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    else:
        milestones = [int(EPOCHS * 0.6), int(EPOCHS * 0.85)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    # ---------- Load checkpoint logic ----------
    start_epoch = 0
    best_map = -1.0

    if LOAD_FROM:  # finetune from a base model (weights only)
        print(f"[INFO] Loading base weights from: {LOAD_FROM}")
        load_ckpt_flex(model, Path(LOAD_FROM), load_optimizer=False)
    if RESUME_FROM:  # full resume (weights + optimizer + epoch)
        print(f"[INFO] Resuming from: {RESUME_FROM}")
        e, best = load_ckpt_flex(model, Path(RESUME_FROM), load_optimizer=True, optimizer=optimizer)
        start_epoch = max(0, e + 1)
        best_map = best

    # ---------- Training loop ----------
    best_map, best_epoch = -1.0, -1
    epochs_no_improve = 0

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_dl, optimizer, DEVICE, scaler)
        scheduler.step()
        print(f"Avg Train Loss: {train_loss:.4f}")

        if (epoch + 1) % EVAL_INTERVAL == 0:
            val_map, _ = evaluate_map(model, val_dl, DEVICE, desc="Eval (val)")
            print(f"Val mAP: {val_map:.4f}")

            # always save last
            save_ckpt(OUT_DIR / "last.pth", model, optimizer, epoch, best_map)

            # improvement check
            if val_map > (best_map + EARLY_STOP_MIN_DELTA):
                best_map, best_epoch = val_map, epoch
                save_ckpt(OUT_DIR / "best.pth", model, optimizer, epoch, best_map)
                print("-> Best model updated.")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"(no improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE})")

                # early stop condition
                if EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping triggered (patience={EARLY_STOP_PATIENCE}).")
                    break

    print(f"Training done. Best val mAP: {best_map:.4f}" + (f" at epoch {best_epoch}." if best_epoch >= 0 else ""))


    # ---------- Final test evaluation ----------
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
