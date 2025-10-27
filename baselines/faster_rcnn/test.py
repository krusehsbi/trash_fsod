#!/usr/bin/env python3
"""
Evaluate mAP@0.5 (VOC-style) for a Faster R-CNN checkpoint trained on a YOLO-format dataset.

Improvements in this version:
- Supports YOLO data.yaml where `train`/`val` are **folders OR .txt files** listing image paths.
- Robust label path resolution by swapping `/images/` -> `/labels/` and replacing the extension with `.txt`.
- Correct class indexing (+1 background for torchvision retained consistently).
- Uses your chosen IoU threshold (default 0.5) for matching, not NMS.
- Clean AP computation per class (one-to-one matching, VOC interpolation).

Usage
-----
python eval_map50.py \
  --data /path/to/yolo_root_or_dataset.yaml \
  --ckpt /path/to/best.pt \
  --batch 2 \
  --imgsz 1024 \
  --iou 0.5 \
  --conf 0.0
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Sequence
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from PIL import Image

# -----------------
# Helper functions
# -----------------

def yolo_denorm_to_xyxy(xc: float, yc: float, w: float, h: float, W: int, H: int):
    x1 = (xc - w/2.0) * W
    y1 = (yc - h/2.0) * H
    x2 = (xc + w/2.0) * W
    y2 = (yc + h/2.0) * H
    return max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)


def label_path_for_image(img_path: Path) -> Path:
    parts = list(img_path.parts)
    try:
        idx = parts.index('images')
        parts[idx] = 'labels'
        label_dir = Path(*parts[:idx+1])
        label_rel = Path(*parts[idx+1:]).with_suffix('.txt')
        return label_dir / label_rel
    except ValueError:
        # No 'images' segment; fallback to sibling labels directory
        return img_path.parent.parent / 'labels' / (img_path.stem + '.txt')


# -----------------
# Dataset wrapper
# -----------------
class YoloDetectionDataset(Dataset):
    def __init__(self, images: Sequence[Path], class_names: List[str], max_size: Optional[int] = None,
                 images_root: Optional[Path] = None, labels_root: Optional[Path] = None):
        self.images = list(images)
        assert len(self.images) > 0, "No images provided to dataset"
        self.class_names = class_names
        self.max_size = max_size
        # optional roots to preserve subfolder structure when resolving label paths
        self.images_root = Path(images_root) if images_root is not None else None
        self.labels_root = Path(labels_root) if labels_root is not None else None

    def __len__(self):
        return len(self.images)

    def _load_targets(self, img_path: Path, W: int, H: int) -> Dict[str, torch.Tensor]:
        # Prefer using provided images_root/labels_root so subfolder structure is preserved
        if self.images_root is not None and self.labels_root is not None:
            try:
                rel = img_path.relative_to(self.images_root)
                label_path = (self.labels_root / rel).with_suffix('.txt')
            except Exception:
                label_path = label_path_for_image(img_path)
        else:
            label_path = label_path_for_image(img_path)
        boxes, raw_labels = [], []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1, y1, x2, y2 = yolo_denorm_to_xyxy(xc, yc, w, h, W, H)
                    boxes.append([x1, y1, x2, y2])
                    raw_labels.append(cls)
        labels = []
        if raw_labels:
            if max(raw_labels) >= len(self.class_names):
                raw_labels = [r - 1 for r in raw_labels]
            labels = [r + 1 for r in raw_labels]  # torchvision: 1..C
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        return {"boxes": boxes, "labels": labels}

    def _resize_keep_ar(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        if self.max_size is None:
            return image, target
        W, H = image.size
        max_side = max(W, H)
        if max_side <= self.max_size:
            return image, target
        scale = self.max_size / max_side
        newW, newH = int(W * scale), int(H * scale)
        image = image.resize((newW, newH), Image.BILINEAR)
        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= (newW / W)
            boxes[:, [1, 3]] *= (newH / H)
            target["boxes"] = boxes
        return image, target

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        W, H = image.size
        target = self._load_targets(img_path, W, H)
        target["image_id"] = torch.tensor([idx])
        image, target = self._resize_keep_ar(image, target)
        image = F.to_tensor(image)
        return image, target


# -----------------
# Data discovery
# -----------------

def parse_yolo_data_yaml(yaml_path: Path):
    data = yaml.safe_load(open(yaml_path))
    names = data.get('names')
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    assert isinstance(names, list), "YOLO data YAML must contain a 'names' list or dict."

    def resolve_path(p):
        p = Path(p)
        return p if p.is_absolute() else (yaml_path.parent / p).resolve()

    train = resolve_path(data['train'])
    val = resolve_path(data.get('val') or data.get('validation') or data.get('valid'))

    def load_split(p: Path) -> List[Path]:
        if p.suffix.lower() == '.txt':
            return [Path(line.strip()) if Path(line.strip()).is_absolute() else (p.parent / line.strip()).resolve()
                    for line in open(p) if line.strip()]
        else:
            # directory
            exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            return sorted([q for q in p.rglob('*') if q.suffix.lower() in exts])

    imgs_train = load_split(train)
    imgs_val = load_split(val)
    return names, imgs_train, imgs_val


# ---------
# Model
# ---------

def build_model(num_classes: int) -> nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


# -----------------
# AP/mAP utilities
# -----------------

def voc_ap(rec: torch.Tensor, prec: torch.Tensor) -> float:
    # Append boundary points
    mrec = torch.cat([torch.tensor([0.0]), rec, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), prec, torch.tensor([0.0])])
    # Precision envelope
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i-1] = torch.maximum(mpre[i-1], mpre[i])
    # Area under curve where recall changes
    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).squeeze()
    return torch.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]).item()


def eval_map50(loader: DataLoader, model: nn.Module, device, iou_thr=0.5, conf_thr=0.0):
    model.eval()
    # Prepare GTs grouped by (img_id, cls)
    gt = {}  # (img_id, cls) -> {boxes: Tensor[N,4], detected: Tensor[N]}
    for img_id, img_path in enumerate(loader.dataset.images):
        # load GT and apply the same resize transform as __getitem__
        image = Image.open(img_path).convert('RGB')
        W, H = image.size
        target = loader.dataset._load_targets(img_path, W, H)
        # if dataset resizes images for evaluation, apply same resize to GT boxes
        if hasattr(loader.dataset, '_resize_keep_ar'):
            image, target = loader.dataset._resize_keep_ar(image, target)
        labels = target['labels']
        boxes = target['boxes']
        for cls_id in labels.unique().tolist() if labels.numel() else []:
            mask = labels == cls_id
            gt[(img_id, int(cls_id))] = {
                'boxes': boxes[mask],
                'detected': torch.zeros((int(mask.sum().item()),), dtype=torch.bool)
            }

    # --- DEBUG: print GT totals / per-class counts
    total_gt_boxes = sum(v['boxes'].shape[0] for v in gt.values())
    per_cls = {}
    for (img_id, cls_id), v in gt.items():
        per_cls[cls_id] = per_cls.get(cls_id, 0) + int(v['boxes'].shape[0])
    print(f"[DBG] GT: total_boxes={total_gt_boxes}, per_class={per_cls}")
    # Collect detections by class: cls -> [(img_id, score, box)]
    dets_by_cls: Dict[int, List[Tuple[int, float, torch.Tensor]]] = {}
    with torch.no_grad():
        first_batch = True
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            # --- DEBUG: print a sample of model outputs for first batch
            if first_batch:
                print("[DBG] Sample model outputs (first batch):")
                for i, (out, tgt) in enumerate(zip(outputs, targets)):
                    img_id = int(tgt['image_id'].item()) if 'image_id' in tgt else i
                    out_boxes = out['boxes'].detach().cpu()
                    out_scores = out['scores'].detach().cpu()
                    out_labels = out['labels'].detach().cpu()
                    print(f"  img_id={img_id}, #pred={len(out_scores)}")
                    if len(out_scores) > 0:
                        topk = min(5, len(out_scores))
                        for k in range(topk):
                            b = out_boxes[k].tolist()
                            s = float(out_scores[k].item())
                            l = int(out_labels[k].item())
                            print(f"    pred{k}: score={s:.3f}, label={l}, box={b}")
                # also print GT for the same first image id (if present)
                try:
                    sample_img_id = int(targets[0]['image_id'].item())
                    sample_gt = {k:v for k,v in gt.items() if k[0]==sample_img_id}
                    print(f"[DBG] GT for sample img_id={sample_img_id}: { {k[1]: v['boxes'].tolist() for k,v in sample_gt.items()} }")
                except Exception:
                    pass
                first_batch = False
            for out, tgt in zip(outputs, targets):
                img_id = int(tgt['image_id'].item())
                boxes = out['boxes'].detach().cpu()
                scores = out['scores'].detach().cpu()
                labels = out['labels'].detach().cpu()  # 1..C
                keep = scores >= conf_thr
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                for b, s, c in zip(boxes, scores, labels):
                    dets_by_cls.setdefault(int(c.item()), []).append((img_id, float(s.item()), b))
    aps = {}
    class_ids = sorted(set([c for c in dets_by_cls.keys()] + [c for (_, c) in [(k[0], k[1]) for k in gt.keys()]]))
    for cls_id in class_ids:
        dets = dets_by_cls.get(cls_id, [])
        # Sort detections by confidence desc
        dets.sort(key=lambda x: x[1], reverse=True)
        tp = []
        fp = []
        npos = sum(gt[(img_id, cls_id)]['boxes'].size(0) for img_id in range(len(loader.dataset)) if (img_id, cls_id) in gt)
        for img_id, score, box in dets:
            if (img_id, cls_id) not in gt or gt[(img_id, cls_id)]['boxes'].numel() == 0:
                tp.append(0); fp.append(1); continue
            gt_boxes = gt[(img_id, cls_id)]['boxes']
            detected = gt[(img_id, cls_id)]['detected']
            ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
            max_iou, j = (ious.max(0))
            if max_iou.item() >= iou_thr and not detected[j]:
                tp.append(1); fp.append(0); detected[j] = True
            else:
                tp.append(0); fp.append(1)
        if npos == 0:
            continue
        tp_c = torch.tensor(tp).cumsum(0)
        fp_c = torch.tensor(fp).cumsum(0)
        rec = tp_c / max(npos, 1)
        prec = tp_c / torch.clamp(tp_c + fp_c, min=1)
        aps[cls_id] = voc_ap(rec, prec)

    map50 = (sum(aps.values()) / len(aps)) if aps else 0.0
    return map50, aps


# -------
#  Main
# -------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to YOLO root or data.yaml')
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint (model.state_dict() or wrapped dict)')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--conf', type=float, default=0.0)
    args = parser.parse_args()

    data_path = Path(args.data)
    if data_path.suffix.lower() in {'.yaml', '.yml'} and data_path.exists():
        class_names, imgs_train, imgs_val = parse_yolo_data_yaml(data_path)
    else:
        root = data_path
        data_yaml = root / 'dataset.yaml'
        if data_yaml.exists():
            class_names, _, imgs_val = parse_yolo_data_yaml(data_yaml)
        else:
            # fallback to default folders
            exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            imgs_val = sorted([q for q in (root/'images/val').rglob('*') if q.suffix.lower() in exts])
            class_names = [str(i) for i in range(1)]

    # try to infer images_root/labels_root from the first validation image path (if possible)
    images_root = None
    labels_root = None
    if len(imgs_val):
        p = imgs_val[0]
        try:
            parts = list(p.parts)
            idx = parts.index('images')
            images_root = Path(*parts[:idx+1])
            labels_root = Path(*parts[:idx]) / 'labels'
        except ValueError:
            # fallback: if imgs_val were produced from a folder like root/'images/val'
            if len(p.parts) >= 3:
                # try to find 'images' segment in parent names
                for i in range(len(p.parts)-1, -1, -1):
                    if p.parts[i] == 'images':
                        images_root = Path(*p.parts[:i+1])
                        labels_root = Path(*p.parts[:i]) / 'labels'
                        break
    ds = YoloDetectionDataset(imgs_val, class_names, max_size=args.imgsz, images_root=images_root, labels_root=labels_root)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=lambda b: tuple(zip(*b)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(num_classes=len(class_names)).to(device)

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)

    # --- Robustly load either a full saved model (nn.Module) or a state_dict
    def _strip_module(sd):
        return {k.replace('module.', ''): v for k, v in sd.items()}

    state_dict = None
    # If user saved the whole model object
    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
        print("[CKPT] Loaded full nn.Module from checkpoint file.")
    elif isinstance(ckpt, dict):
        # If checkpoint contains a model object under common keys
        for obj_key in ('model', 'net', 'module', 'detector'):
            if obj_key in ckpt and isinstance(ckpt[obj_key], nn.Module):
                model = ckpt[obj_key].to(device)
                print(f"[CKPT] Loaded full nn.Module from checkpoint['{obj_key}'].")
                break
        else:
            # Try to find a state_dict inside
            for key in ('model_state', 'state_dict', 'model_state_dict', 'model'):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = _strip_module(ckpt[key])
                    break
            # If the ckpt dict itself is a raw state_dict (tensor values)
            if state_dict is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = _strip_module(ckpt)
            # classes saved
            ckpt_classes = ckpt.get('classes') or ckpt.get('labels') or ckpt.get('class_names')
            if state_dict is None:
                # nothing usable found as state dict and no model object loaded
                raise RuntimeError(f"Couldn't find model object or state_dict inside checkpoint: {ckpt_path}")
            # rebuild head using saved classes if present
            if ckpt_classes is not None:
                model = build_model(num_classes=len(ckpt_classes)).to(device)
                class_names_for_eval = ckpt_classes
            else:
                class_names_for_eval = class_names
            # load state dict (handle backbone-only checkpoints)
            sd_keys = list(state_dict.keys())
            if any(k.startswith('backbone.body.') for k in sd_keys) and not any('roi_heads' in k or 'box_predictor' in k for k in sd_keys):
                # backbone-only -> strip prefix and load into backbone.body
                sd_back = {k.replace('backbone.body.', ''): v for k, v in state_dict.items()}
                missing, unexpected = model.backbone.body.load_state_dict(sd_back, strict=False)
                print("[CKPT] Loaded backbone-only weights into model.backbone.body. missing:", missing, "unexpected:", unexpected)
            else:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print('[CKPT] load_state_dict missing:', missing, 'unexpected:', unexpected)
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    # Quick parameter sanity checks (helps detect uninitialized head)
    try:
        wmean = float(model.roi_heads.box_predictor.cls_score.weight.mean().cpu().item())
        bmean = float(model.roi_heads.box_predictor.cls_score.bias.mean().cpu().item())
        print(f"[CKPT] cls_score.weight.mean={wmean:.6f}, bias.mean={bmean:.6f}")
    except Exception as e:
        print("[CKPT] could not inspect head params:", e)

    # Run a tiny forward on the first dataset image to verify model produces detections
    try:
        sample_imgs, sample_tgts = next(iter(loader))
        sample_imgs = [img.to(device) for img in sample_imgs[:1]]
        model.eval()
        with torch.no_grad():
            sample_out = model(sample_imgs)
        print(f"[CKPT] sample_out[0] boxes={len(sample_out[0].get('boxes', []))}, scores={sample_out[0].get('scores', [])[:5]}")

        # --- EXTRA DEBUG: inspect RPN / ROI intermediate outputs (robust)
        try:
            img_list = model.transform(sample_imgs)
            # some torchvision versions return (images, targets)
            if isinstance(img_list, tuple):
                img_list = img_list[0]
            tensors = getattr(img_list, 'tensors', img_list)
            image_sizes = getattr(img_list, 'image_sizes', None)

            # backbone features
            features = model.backbone(tensors)
            if isinstance(features, torch.Tensor):
                features = {'0': features}
            print("[CKPT] backbone feature keys:", list(features.keys()))
            for k, v in features.items():
                print(f"  feat {k} shape: {tuple(v.shape)}")

            # RPN -> proposals
            proposals, _ = model.rpn(img_list, features, None)
            print(f"[CKPT] RPN proposals per image: {[len(p) for p in proposals]}")
            if len(proposals) and proposals[0].numel():
                print(f"[CKPT] sample proposals (first image, first 5): {proposals[0][:5].tolist()}")

            # roi_heads raw outputs (before postprocess)
            try:
                raw_dets = model.roi_heads(features, proposals, image_sizes, None)
                print(f"[CKPT] roi_heads raw detections per image: {[len(d) for d in raw_dets]}")
            except Exception as e2:
                print("[CKPT] roi_heads call failed:", e2)
        except Exception as e:
            print("[CKPT] internal RPN/ROI debug failed:", e)
    except Exception as e:
        print("[CKPT] forward check failed:", e)

    # After loading checkpoint/state_dict into `model`:
    # Ensure torchvision postprocess uses the requested confidence threshold
    try:
        # torchvision versions differ in attribute names
        if hasattr(model, 'roi_heads') and hasattr(model.roi_heads, 'score_thresh'):
            model.roi_heads.score_thresh = args.conf
            print(f"[CKPT] set roi_heads.score_thresh = {args.conf}")
        else:
            # older/newer variants
            if hasattr(model, 'box_score_thresh'):
                model.box_score_thresh = args.conf
                print(f"[CKPT] set model.box_score_thresh = {args.conf}")
            if hasattr(model.roi_heads, 'box_score_thresh'):
                model.roi_heads.box_score_thresh = args.conf
                print(f"[CKPT] set model.roi_heads.box_score_thresh = {args.conf}")
    except Exception as e:
        print("[CKPT] could not set score threshold:", e)

    # Extra debug: print raw roi_heads outputs (inspect scores/labels before postprocess)
    # This helps confirm whether roi_heads produces detections but they are filtered out.
    try:
        sample_imgs, _ = next(iter(loader))
        sample_imgs = [img.to(device) for img in sample_imgs[:1]]
        model.eval()
        with torch.no_grad():
            # run transform + backbone + RPN -> proposals
            img_list = model.transform(sample_imgs)
            if isinstance(img_list, tuple):
                img_list = img_list[0]
            tensors = getattr(img_list, 'tensors', img_list)
            image_sizes = getattr(img_list, 'image_sizes', None)

            features = model.backbone(tensors)
            if isinstance(features, torch.Tensor):
                features = {'0': features}

            proposals, _ = model.rpn(img_list, features, None)

            raw_dets = model.roi_heads(features, proposals, image_sizes, None)

            print("[CKPT] raw roi_heads outputs:")
            for i, rd in enumerate(raw_dets):
                if isinstance(rd, dict):
                    print(f"  img[{i}] keys: {list(rd.keys())}")
                    for k, v in rd.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={tuple(v.shape)}, min={float(v.min()):.6f}, max={float(v.max()):.6f}")
                        else:
                            print(f"    {k}: type={type(v)}")
                elif isinstance(rd, (list, tuple)):
                    print(f"  img[{i}] tuple len={len(rd)}")
                    for j, elem in enumerate(rd):
                        if isinstance(elem, torch.Tensor):
                            print(f"    elem{j}: shape={tuple(elem.shape)}, min={float(elem.min()):.6f}, max={float(elem.max()):.6f}")
                        else:
                            print(f"    elem{j}: type={type(elem)}")
                else:
                    print(f"  img[{i}] unknown raw_dets type: {type(rd)}")
    except Exception as e:
        print("[CKPT] raw roi_heads debug failed:", e)

    map50, aps = eval_map50(loader, model, device, iou_thr=args.iou, conf_thr=args.conf)

    print(f"mAP@{args.iou:.2f}: {map50:.4f}")
    if len(aps) == 0:
        # Extra debugging hints
        num_gt = 0
        for img_path in loader.dataset.images:
            with Image.open(img_path) as im:
                W, H = im.size
            t = loader.dataset._load_targets(img_path, W, H)
            num_gt += int(t['labels'].numel())
        print(f"[DEBUG] No APs computed. GT boxes in val set: {num_gt}.")
        if num_gt == 0:
            print("[CAUSE] No ground-truth boxes found for validation. Check your labels/val path mapping.")
        else:
            print("[CAUSE] Model may not be loading the trained head (class mismatch). If training saved 'classes', this script now uses them to rebuild the head.")
    for cls_id, ap in sorted(aps.items()):
        name = class_names[cls_id - 1] if 1 <= cls_id <= len(class_names) else str(cls_id)
        print(f"  class {cls_id:2d} ({name}): AP@{args.iou:.2f} = {ap:.4f}")

if __name__ == '__main__':
    main()
