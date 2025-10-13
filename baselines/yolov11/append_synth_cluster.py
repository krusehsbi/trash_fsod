#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append g *cluster-center* synthetic YOLO instances per class to an existing few-shot YOLO dataset.

Pipeline (per class):
  1) Collect ALL object instances from the synthetic set (labels + matching images).
  2) Crop object areas (bbox + padding), compute CLIP embeddings.
  3) K-Means into g clusters (or fewer if not enough instances).
  4) Select the most central instance (closest to centroid) from each cluster.
  5) Copy the corresponding (image, label) pairs into the output train set.

Notes:
- Works with nested folders for images/labels in the synthetic dataset.
- Keeps val/test from the few-shot dataset unchanged.
"""

import os
import argparse
import shutil
from glob import glob
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
from PIL import Image
from tqdm import tqdm

# Torch / CLIP
import torch
import clip

# Clustering
from sklearn.cluster import KMeans

# -------------------------
# Data structures
# -------------------------
@dataclass
class InstanceRef:
    cls: int
    label_path: str
    image_path: str
    line_idx: int          # which line in the label file
    bbox_xyxy: Tuple[int, int, int, int]  # pixel coords (x1, y1, x2, y2)


# -------------------------
# Helpers
# -------------------------
def yolo_to_xyxy(yolo_box: Tuple[float, float, float, float], W: int, H: int) -> Tuple[int, int, int, int]:
    """Convert YOLO (cx, cy, w, h) normalized to pixel (x1, y1, x2, y2)."""
    cx, cy, w, h = yolo_box
    cx, cy, w, h = cx * W, cy * H, w * W, h * H
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))
    return x1, y1, x2, y2


def pad_box(x1, y1, x2, y2, W, H, pad_px: int):
    """Pad box by pad_px pixels on all sides, clamped to image."""
    x1p = max(0, x1 - pad_px)
    y1p = max(0, y1 - pad_px)
    x2p = min(W - 1, x2 + pad_px)
    y2p = min(H - 1, y2 + pad_px)
    return x1p, y1p, x2p, y2p


def find_image_for_label(synth_img_root: str, name: str, exts=(".jpg", ".jpeg", ".png")) -> Optional[str]:
    """Find matching image for a label filename stem, searching recursively."""
    # Common case: flat or mirrored structure
    for ext in exts:
        candidate = os.path.join(synth_img_root, name + ext)
        if os.path.exists(candidate):
            return candidate
    # Fallback: recursive search
    found = []
    for ext in exts:
        found += glob(os.path.join(synth_img_root, "**", f"{name}{ext}"), recursive=True)
    return found[0] if found else None


def parse_instances_from_label(label_path: str, image_path: str, pad_ratio: float = 0.05) -> List[InstanceRef]:
    """Parse all instances from a YOLO label file and return pixel-space boxes with padding."""
    insts: List[InstanceRef] = []
    if not os.path.exists(label_path) or not os.path.exists(image_path):
        return insts

    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            W, H = im.size
    except Exception:
        return insts

    pad_px = int(round(pad_ratio * max(W, H)))

    with open(label_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except Exception:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy((cx, cy, w, h), W, H)
            x1p, y1p, x2p, y2p = pad_box(x1, y1, x2, y2, W, H, pad_px)
            insts.append(
                InstanceRef(
                    cls=cls,
                    label_path=label_path,
                    image_path=image_path,
                    line_idx=i,
                    bbox_xyxy=(x1p, y1p, x2p, y2p),
                )
            )
    return insts


def copy_tree(src, dst):
    """Recursively copy directory (images/ or labels/) while preserving structure."""
    if not os.path.exists(src):
        return
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        dest_dir = os.path.join(dst, rel)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            src_path = os.path.join(root, f)
            dst_path = os.path.join(dest_dir, f)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)


# -------------------------
# CLIP embedding
# -------------------------
class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_crops(self, crops: List[Image.Image], batch_size: int = 64) -> np.ndarray:
        """Return L2-normalized image embeddings for a list of PIL images."""
        embs = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i : i + batch_size]
            tens = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
            feats = self.model.encode_image(tens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.cpu().numpy())
        if not embs:
            return np.zeros((0, 512), dtype=np.float32)
        return np.concatenate(embs, axis=0).astype(np.float32)


# -------------------------
# Core logic
# -------------------------
def cluster_select_instances(
    instances: List[InstanceRef],
    embedder: ClipEmbedder,
    g_per_class: int,
    random_state: int = 0,
) -> List[InstanceRef]:
    """Cluster instances (per class already filtered) and select the most central instance per cluster."""
    if len(instances) == 0:
        return []

    # Load crops
    crops = []
    valid_refs = []
    for inst in instances:
        try:
            with Image.open(inst.image_path) as im:
                im = im.convert("RGB")
                x1, y1, x2, y2 = inst.bbox_xyxy
                # clamp (safety)
                x1 = max(0, min(x1, im.width - 1))
                y1 = max(0, min(y1, im.height - 1))
                x2 = max(0, min(max(x1 + 1, x2), im.width))
                y2 = max(0, min(max(y1 + 1, y2), im.height))
                crop = im.crop((x1, y1, x2, y2))
                crops.append(crop)
                valid_refs.append(inst)
        except Exception:
            continue

    if not crops:
        return []

    # Embed
    X = embedder.encode_crops(crops)
    k = min(g_per_class, len(valid_refs))
    if k <= 0:
        return []

    # K-Means
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    km.fit(X)
    labels = km.labels_
    centers = km.cluster_centers_

    # Pick closest to each centroid
    selected: List[InstanceRef] = []
    for c in range(k):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        Xc = X[idxs]
        ctr = centers[c]
        dists = np.linalg.norm(Xc - ctr, axis=1)
        best_local = idxs[np.argmin(dists)]
        selected.append(valid_refs[best_local])

    return selected


def add_clustered_g_synthetic(
    fewshot_dir: str,
    synth_dir: str,
    out_dir: str,
    g_per_class: int = 10,
    pad_ratio: float = 0.05,
    device: Optional[str] = None,
    model_name: str = "ViT-B/32",
    seed: int = 0,
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
):
    rng = np.random.RandomState(seed)

    few_train_img = os.path.join(fewshot_dir, "images/train")
    few_train_lbl = os.path.join(fewshot_dir, "labels/train")
    synth_img = os.path.join(synth_dir, "images")
    synth_lbl = os.path.join(synth_dir, "labels")

    out_img = os.path.join(out_dir, "images/train")
    out_lbl = os.path.join(out_dir, "labels/train")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    # 1) Copy existing few-shot train
    print("Copying existing few-shot train set...")
    copy_tree(few_train_img, out_img)
    copy_tree(few_train_lbl, out_lbl)

    # 2) Build instance index for synthetic set
    print("Indexing synthetic instances...")
    label_files = glob(os.path.join(synth_lbl, "**", "*.txt"), recursive=True)
    all_instances_by_class: Dict[int, List[InstanceRef]] = defaultdict(list)

    for lbl in tqdm(label_files, desc="Parsing labels"):
        name = os.path.splitext(os.path.basename(lbl))[0]
        img_path = find_image_for_label(synth_img, name, exts=exts)
        if not img_path:
            continue
        insts = parse_instances_from_label(lbl, img_path, pad_ratio=pad_ratio)
        for inst in insts:
            all_instances_by_class[inst.cls].append(inst)

    # 3) Prepare CLIP
    print("Loading CLIP...")
    embedder = ClipEmbedder(model_name=model_name, device=device)

    # 4) For each class, cluster & select g centers
    print("Clustering per class and selecting center instances...")
    selected_instances: List[InstanceRef] = []
    for cls, insts in all_instances_by_class.items():
        if len(insts) == 0:
            continue
        # Shuffle deterministically before embedding (just to vary tie-breaks)
        rng.shuffle(insts)
        picks = cluster_select_instances(insts, embedder, g_per_class=g_per_class, random_state=seed)
        print(f"  • class {cls}: {len(picks)} selected (from {len(insts)} instances)")
        selected_instances.extend(picks)

    # 5) Copy the unique label/img pairs for the selected instances
    print("Copying selected synthetic samples...")
    selected_label_paths: Set[str] = set()
    selected_image_paths: Set[str] = set()
    for inst in selected_instances:
        selected_label_paths.add(inst.label_path)
        selected_image_paths.add(inst.image_path)

    # Copy labels
    for lbl_path in selected_label_paths:
        dst_lbl_path = os.path.join(out_lbl, os.path.basename(lbl_path))
        os.makedirs(os.path.dirname(dst_lbl_path), exist_ok=True)
        shutil.copy2(lbl_path, dst_lbl_path)

    # Copy images
    for img_path in selected_image_paths:
        dst_img_path = os.path.join(out_img, os.path.basename(img_path))
        shutil.copy2(img_path, dst_img_path)

    # 6) Copy val/test unchanged
    for split in ["val", "test"]:
        for sub in ["images", "labels"]:
            src = os.path.join(fewshot_dir, sub, split)
            dst = os.path.join(out_dir, sub, split)
            print(f"Copying {split} {sub}...")
            copy_tree(src, dst)

    print("\nDone!")
    print(f"New dataset saved to: {out_dir}")
    print(f"  images/train → {out_img}")
    print(f"  labels/train → {out_lbl}")


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Append g synthetic YOLO samples per class by CLIP-cluster center selection."
    )
    ap.add_argument("--fewshot-dir", required=True, help="Path to YOLO few-shot dataset root")
    ap.add_argument("--synthetic-dir", required=True, help="Path to YOLO synthetic dataset root")
    ap.add_argument("--out-dir", required=True, help="Output merged dataset path")
    ap.add_argument("--g", type=int, default=10, help="Synthetic instances per class (clusters)")
    ap.add_argument("--pad", type=float, default=0.05, help="Padding ratio around bbox (e.g., 0.05)")
    ap.add_argument("--model", type=str, default="ViT-B/32", help="CLIP model variant")
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (auto if None)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exts", type=str, nargs="+", default=[".jpg", ".jpeg", ".png"], help="Image extensions to try")
    args = ap.parse_args()

    add_clustered_g_synthetic(
        fewshot_dir=args.fewshot_dir,
        synth_dir=args.synthetic_dir,
        out_dir=args.out_dir,
        g_per_class=args.g,
        pad_ratio=args.pad,
        device=args.device,
        model_name=args.model,
        seed=args.seed,
        exts=tuple(args.exts),
    )


if __name__ == "__main__":
    main()
