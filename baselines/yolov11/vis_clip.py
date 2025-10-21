#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot CLIP embeddings of all object instances from a YOLO dataset, with options to keep only certain classes and respect subfolders.
"""

import argparse
import os
import sys
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans

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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def read_names(root: Path) -> Optional[List[str]]:
    names_txt = root / "names.txt"
    if names_txt.exists():
        return [line.strip() for line in names_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    return None

def yolo_txt_to_instances(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    instances = []
    if not txt_path.exists():
        return instances
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            instances.append((cls, cx, cy, w, h))
        except Exception:
            continue
    return instances

def norm_to_xyxy(cx, cy, w, h, W, H, pad_frac=0.0):
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    if pad_frac > 0:
        pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
        x1, y1, x2, y2 = x1 - pw, y1 - ph, x2 + pw, y2 + ph
    x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)])
    return x1, y1, x2, y2

def find_image_for_label(label_path: Path, images_root: Path, labels_root: Path, respect_subfolders: bool) -> Optional[Path]:
    """Find corresponding image for a label file, respecting subfolder structure if requested."""
    rel = label_path.relative_to(labels_root)
    stem = rel.with_suffix("")
    if respect_subfolders:
        for ext in IMG_EXTS:
            candidate = images_root / (str(stem) + ext)
            if candidate.exists():
                return candidate
    else:
        for ext in IMG_EXTS:
            candidate = images_root / (stem.name + ext)
            if candidate.exists():
                return candidate
    return None

def collect_instances(root: Path, images_subdir: str, labels_subdir: str, split: Optional[str], keep_classes: Optional[List[int]], respect_subfolders: bool) -> List[Dict]:
    images_root = root / images_subdir
    labels_root = root / labels_subdir
    if split:
        images_root = images_root / split
        labels_root = labels_root / split

    label_files = [p for p in labels_root.rglob("*.txt") if p.is_file()]
    items = []
    for lp in tqdm(label_files, desc="Scanning labels"):
        img_path = find_image_for_label(lp, images_root, labels_root, respect_subfolders)
        if img_path is None:
            continue
        instances = yolo_txt_to_instances(lp)
        if keep_classes is not None:
            instances = [i for i in instances if i[0] in keep_classes]
        if not instances:
            continue
        items.append(dict(image=img_path, label=lp, instances=instances))
    return items

def load_clip(model_name: str, device: str):
    if clip is None:
        raise RuntimeError("CLIP package not found.")
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
    return model, preprocess

def embed_instances(items: List[Dict], model, preprocess, device: str, max_instances: int, pad_frac: float):
    embs, metas = [], []
    with torch.no_grad():
        for item in tqdm(items, desc="Embedding instances"):
            img = Image.open(item["image"]).convert("RGB")
            W, H = img.size
            for (cls, cx, cy, w, h) in item["instances"]:
                if len(metas) >= max_instances:
                    break
                x1, y1, x2, y2 = norm_to_xyxy(cx, cy, w, h, W, H, pad_frac)
                crop = img.crop((x1, y1, x2, y2))
                image_input = preprocess(crop).unsqueeze(0).to(device)
                feat = model.encode_image(image_input)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                embs.append(feat.cpu())
                # store bbox so we can save *cropped* images later
                metas.append({"image": str(item["image"]), "class": int(cls), "bbox": (x1, y1, x2, y2)})
            if len(metas) >= max_instances:
                break
    if not embs:
        raise RuntimeError("No instances found.")
    return torch.cat(embs).numpy().astype(np.float32), metas

def reduce_to_2d(X, method, seed):
    if X.shape[1] <= 2:
        return X[:, :2]
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    elif method == "tsne":
        return TSNE(n_components=2, random_state=seed).fit_transform(X)
    elif method == "umap":
        if not HAVE_UMAP:
            raise RuntimeError("UMAP not installed.")
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    else:
        raise ValueError("Unknown reduction method.")

def build_colors(num_classes: int):
    import matplotlib.pyplot as plt
    color_list = [d.get('color') for d in plt.rcParams['axes.prop_cycle']]
    while len(color_list) < num_classes:
        color_list += color_list
    return color_list[:num_classes]

def plot_embeddings(Z, metas, out_path, class_names, title, cluster_labels=None):
    import matplotlib.pyplot as plt
    classes = np.array([m["class"] for m in metas], int)
    num_classes = int(classes.max()) + 1 if classes.size else 0
    colors = build_colors(num_classes)

    plt.figure(figsize=(10, 8))
    for c in range(num_classes):
        mask = classes == c
        label = class_names[c] if (class_names and c < len(class_names)) else f"class {c}"
        plt.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.8, label=label)
    # Overlay cluster “circles” if requested
    ax = plt.gca()
    if cluster_labels is not None:
        _draw_cluster_ellipses(ax, Z, cluster_labels, n_std=2.0, lw=2.0)
    #plt.legend(frameon=False)
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2"); plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

def _draw_cluster_ellipses(ax, Z, labels, n_std=2.0, lw=2.0):
    """Draw Gaussian ellipses around each cluster."""
    import numpy as np
    from matplotlib.patches import Ellipse
    for k in np.unique(labels):
        pts = Z[labels == k]
        if pts.shape[0] < 2:
            continue
        mu = pts.mean(axis=0)
        # covariance + eigendecomp for ellipse params
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        # width/height = 2 * n_std * sqrt(eigvals)
        width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        e = Ellipse(xy=mu, width=width, height=height, angle=angle,
                    facecolor="none", edgecolor="black", linewidth=lw, alpha=0.9)
        ax.add_patch(e)
        ax.text(mu[0], mu[1], f"C{k}", ha="center", va="center",
                fontsize=9, weight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.6, alpha=0.7))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--images-subdir", default="images")
    p.add_argument("--labels-subdir", default="labels")
    p.add_argument("--split", default=None)
    p.add_argument("--model", default="ViT-B/32")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reduction", choices=["pca", "tsne", "umap"], default="pca")
    p.add_argument("--max-instances", type=int, default=10000)
    p.add_argument("--pad-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", default="clip_embeddings.png")
    p.add_argument("--classes", nargs="*", type=int, default=None, help="List of class IDs to keep.")
    p.add_argument("--respect-subfolders", action="store_true", help="Respect original subfolder structure when locating images.")
    p.add_argument("--cluster", type=int, default=0, metavar="K",
                   help="If > 0, run K-means with K clusters on the 2D embedding and circle each cluster.")
    return p.parse_args()

def main():
    args = parse_args()
    root = Path(args.root)

    # --- Collect and embed instances ---
    items = collect_instances(
        root, args.images_subdir, args.labels_subdir,
        args.split, args.classes, args.respect_subfolders
    )
    if not items:
        print("No items found.")
        sys.exit(1)

    model, preprocess = load_clip(args.model, args.device)
    X, metas = embed_instances(items, model, preprocess, args.device, args.max_instances, args.pad_frac)
    Z = reduce_to_2d(X, args.reduction, args.seed)
    class_names = read_names(root)

    # --- Optional clustering ---
    cluster_labels = None
    if args.cluster and args.cluster > 0:
        kmeans = KMeans(n_clusters=args.cluster, random_state=args.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(Z)

    # --- Plot embedding scatter (with ellipses) ---
    plot_embeddings(
        Z, metas, Path(args.output), class_names,
        "Distribution of CLIP-Embeddings for Class `Plastic bag & wrapper`",
        cluster_labels=cluster_labels
    )

    # --- Print file names grouped by cluster ---
    if cluster_labels is not None:
        print("\n=== Clustered file names ===")
        from collections import defaultdict
        cluster_to_files = defaultdict(list)
        for lbl, meta in zip(cluster_labels, metas):
            path = meta.get("image")
            cluster_to_files[int(lbl)].append(str(path))

        for k in sorted(cluster_to_files):
            print(f"--- Cluster {k} ---")
            for fname in sorted(cluster_to_files[k]):
                print(f"  {fname}")

    # --- Save a few random cropped examples per cluster ---
    if cluster_labels is not None:
        from PIL import Image
        import random
        rng = random.Random(args.seed)
        n_samples = 10  # how many examples to save per cluster

        out_dir = Path(args.output).with_name(Path(args.output).stem + "_clusters")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving up to {n_samples} cropped examples per cluster in: {out_dir}")

        # group indices by cluster
        from collections import defaultdict
        cluster_to_indices = defaultdict(list)
        for i, lbl in enumerate(cluster_labels):
            cluster_to_indices[int(lbl)].append(i)

        for k, idxs in sorted(cluster_to_indices.items()):
            if not idxs:
                continue
            chosen = rng.sample(idxs, min(n_samples, len(idxs)))
            cluster_dir = out_dir / f"cluster_{k}"
            cluster_dir.mkdir(exist_ok=True)

            for j, i in enumerate(chosen):
                meta = metas[i]
                img_path = Path(meta["image"])
                bbox = meta.get("bbox")
                if not bbox:
                    print(f"  [warn] no bbox for {img_path}, skipping")
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"  [warn] failed to open {img_path}: {e}")
                    continue
                crop = img.crop((x1, y1, x2, y2))
                save_name = f"{img_path.stem}_c{k}_{j}.jpg"
                crop.save(cluster_dir / save_name, quality=95)

        print("Done saving cropped examples.")




if __name__ == "__main__":
    main()