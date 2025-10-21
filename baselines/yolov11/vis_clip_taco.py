#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot CLIP embeddings of all object instances from a COCO dataset.
Uses supercategories as class labels, with support for comma-separated supercategory input.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

try:
    import clip
except Exception:
    clip = None

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

def coco_to_xyxy(bbox, W, H, pad_frac=0.0, min_size: int = 1):
    """Robustly convert COCO xywh to integer xyxy within image bounds.
    Returns (x1, y1, x2, y2) or None if the box is invalid/degenerate.
    """
    import math
    x, y, w, h = bbox
    # Reject non-finite or non-positive sizes
    vals = [x, y, w, h]
    if any((v is None) for v in vals):
        return None
    try:
        x, y, w, h = float(x), float(y), float(w), float(h)
    except Exception:
        return None
    if not np.isfinite([x, y, w, h]).all():
        return None
    if w <= 0 or h <= 0:
        return None

    x1, y1, x2, y2 = x, y, x + w, y + h

    # Optional padding
    if pad_frac > 0:
        pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
        x1, y1, x2, y2 = x1 - pw, y1 - ph, x2 + pw, y2 + ph

    # Intersect with image bounds
    x1 = max(0.0, min(x1, W - 1.0))
    y1 = max(0.0, min(y1, H - 1.0))
    x2 = max(0.0, min(x2, W - 1.0))
    y2 = max(0.0, min(y2, H - 1.0))

    # Ensure valid ordering
    if x2 <= x1 or y2 <= y1:
        return None

    # Integer pixel box
    x1i = int(math.floor(x1))
    y1i = int(math.floor(y1))
    x2i = int(math.ceil(x2))
    y2i = int(math.ceil(y2))

    # Enforce minimum size
    if (x2i - x1i) < min_size:
        x2i = min(W - 1, x1i + min_size)
    if (y2i - y1i) < min_size:
        y2i = min(H - 1, y1i + min_size)

    if x2i <= x1i or y2i <= y1i:
        return None

    return x1i, y1i, x2i, y2i

def load_coco_annotations(coco_json: Path, keep_supers: Optional[List[str]] = None):
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    img_info = {im['id']: im for im in coco['images']}
    cat_info = {c['id']: c for c in coco['categories']}

    anns_by_img = {}
    for ann in coco['annotations']:
        cat = cat_info[ann['category_id']]
        supercat = cat.get('supercategory', 'unknown')
        if keep_supers and supercat not in keep_supers:
            continue
        ann['supercategory'] = supercat
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    supercats = sorted(list({c.get('supercategory', 'unknown') for c in cat_info.values()}))
    super2id = {sc: i for i, sc in enumerate(supercats)}

    return img_info, anns_by_img, super2id

def load_clip(model_name: str, device: str):
    if clip is None:
        raise RuntimeError('CLIP not installed. Install via `pip install git+https://github.com/openai/CLIP.git`')
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
    return model, preprocess

def embed_instances(images_root: Path, img_info: Dict, anns_by_img: Dict, model, preprocess, device: str, pad_frac: float, max_instances: int, respect_subfolders: bool, super2id: Dict):
    embs, metas = [], []
    with torch.no_grad():
        for img_id, anns in tqdm(anns_by_img.items(), desc='Embedding instances'):
            info = img_info[img_id]
            fname = info['file_name']
            img_path = images_root / fname if respect_subfolders else images_root / os.path.basename(fname)
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert('RGB')
            W, H = img.size

            for ann in anns:
                if len(metas) >= max_instances:
                    break
                box = coco_to_xyxy(ann['bbox'], W, H, pad_frac)
                if box is None:
                    # Skip invalid/degenerate boxes to avoid PIL crop errors
                    continue
                x1, y1, x2, y2 = box
                crop = img.crop((x1, y1, x2, y2))
                image_input = preprocess(crop).unsqueeze(0).to(device)
                feat = model.encode_image(image_input)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                embs.append(feat.cpu())
                metas.append({'image': str(img_path), 'supercategory': ann['supercategory'], 'super_id': super2id.get(ann['supercategory'], -1)})
            if len(metas) >= max_instances:
                break
    if not embs:
        raise RuntimeError('No instances found.')
    return torch.cat(embs).numpy().astype(np.float32), metas

def reduce_to_2d(X, method, seed):
    if X.shape[1] <= 2:
        return X[:, :2]
    if method == 'pca':
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    elif method == 'tsne':
        return TSNE(n_components=2, random_state=seed).fit_transform(X)
    elif method == 'umap':
        if not HAVE_UMAP:
            raise RuntimeError('UMAP not installed.')
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    else:
        raise ValueError('Unknown reduction method.')

def build_colors(num_classes: int):
    import matplotlib.pyplot as plt
    colors = [d.get('color') for d in plt.rcParams['axes.prop_cycle']]
    while len(colors) < num_classes:
        colors += colors
    return colors[:num_classes]

def plot_embeddings(Z, metas, out_path: Path, supercats: List[str], title: str):
    import matplotlib.pyplot as plt
    classes = np.array([m['super_id'] for m in metas], int)
    num_classes = int(classes.max()) + 1 if classes.size else 0
    colors = build_colors(num_classes)

    plt.figure(figsize=(10, 8))
    for c in range(num_classes):
        mask = classes == c
        label = supercats[c] if c < len(supercats) else f'class {c}'
        plt.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.8, label=label)
    #plt.legend(frameon=False)
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2'); plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f'Saved plot to {out_path}')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--coco-json', required=True, help='Path to COCO annotations JSON')
    p.add_argument('--images-dir', required=True, help='Path to image directory')
    p.add_argument('--model', default='ViT-B/32', help='CLIP model')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--reduction', choices=['pca', 'tsne', 'umap'], default='pca')
    p.add_argument('--max-instances', type=int, default=10000)
    p.add_argument('--pad-frac', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--output', default='clip_embeddings_coco.png')
    p.add_argument('--supercategories', type=str, default=None, help='Comma-separated list of supercategories to keep')
    p.add_argument('--respect-subfolders', action='store_true', help='Respect subfolder structure in file names')
    return p.parse_args()

def main():
    args = parse_args()
    supercats = None
    if args.supercategories:
        supercats = [s.strip() for s in args.supercategories.split(',') if s.strip()]

    coco_json = Path(args.coco_json)
    images_dir = Path(args.images_dir)

    img_info, anns_by_img, super2id = load_coco_annotations(coco_json, supercats)
    model, preprocess = load_clip(args.model, args.device)

    X, metas = embed_instances(images_dir, img_info, anns_by_img, model, preprocess, args.device, args.pad_frac, args.max_instances, args.respect_subfolders, super2id)
    Z = reduce_to_2d(X, args.reduction, args.seed)

    supercat_list = list(super2id.keys())
    plot_embeddings(Z, metas, Path(args.output), supercat_list, 'COCO Instance CLIP Embeddings (Supercategories)')

if __name__ == '__main__':
    main()
