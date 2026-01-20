#!/usr/bin/env python3
"""
Make the "Average bbox sizes by category and respective AP" bubble plot.

What it does:
1) Reads a YOLO-style validation set (images + .txt labels with normalized xywh).
2) Computes per-class average bbox width/height in pixels + count of GT boxes.
3) Evaluates a trained Ultralytics RT-DETR model on the same split to get per-class AP.
4) Plots:
   - X = avg bbox width (px)
   - Y = avg bbox height (px)
   - bubble size = GT count (sqrt-scaled)
   - color = per-class AP

Requires:
- opencv-python
- numpy
- matplotlib
- pyyaml
- ultralytics  (for RT-DETR evaluation)

Usage examples:
  python plot_bbox_ap.py --weights runs/detect/train/weights/best.pt --data data.yaml
  python plot_bbox_ap.py --weights best.pt --data data.yaml --split val --imgsz 640 --ap-scale 100 --out plot.png
  python plot_bbox_ap.py --weights best.pt --data data.yaml --images /path/images/val --labels /path/labels/val
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _as_path(p):
    return Path(p).expanduser().resolve()


def _resolve_split_path(data_dict, split_key: str) -> str:
    """
    Ultralytics-style data.yaml can be:
      path: /dataset/root
      train: images/train
      val: images/val
    or
      train: /abs/path/to/images/train
      val: /abs/path/to/images/val

    Returns absolute-ish string path to split images dir, if present.
    """
    if split_key not in data_dict:
        raise KeyError(f"'{split_key}' not found in data yaml. Found keys: {list(data_dict.keys())}")

    base = data_dict.get("path", None)
    split_val = data_dict[split_key]

    # split_val can be list for multi-source datasets; we handle the common single-path case
    if isinstance(split_val, (list, tuple)):
        if len(split_val) != 1:
            raise ValueError(f"Expected a single path for '{split_key}', got list: {split_val}")
        split_val = split_val[0]

    # If it's already absolute, use it; else join with base path if provided
    split_path = Path(split_val)
    if not split_path.is_absolute() and base is not None:
        split_path = Path(base) / split_path

    return str(split_path)


def _default_labels_dir_from_images(images_dir: str) -> str:
    """
    Try common YOLO conventions:
      .../images/val  -> .../labels/val
      .../images      -> .../labels
    If neither matches, fallback to sibling folder named 'labels' at same level.
    """
    p = Path(images_dir)
    parts = list(p.parts)

    # Replace an 'images' component with 'labels'
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")  # last occurrence
        parts[i] = "labels"
        return str(Path(*parts))

    # Otherwise sibling 'labels'
    if p.parent.exists():
        return str(p.parent / "labels")

    return str(p / "labels")


def load_yolo_gt_stats(images_dir: str, labels_dir: str, class_names=None):
    """
    Reads YOLO txt labels and computes per-class:
      - avg bbox width in px
      - avg bbox height in px
      - count of GT bboxes

    YOLO format (per line):
      class x_center y_center width height   (all normalized to [0,1])
    """
    images_dir = _as_path(images_dir)
    labels_dir = _as_path(labels_dir)

    # Map image stem -> image path
    img_map = {}
    for ext in IMG_EXTS:
        for p in images_dir.rglob(f"*{ext}"):
            img_map[p.stem] = p

    sum_w = {}
    sum_h = {}
    count = {}

    label_files = list(labels_dir.rglob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No .txt labels found under: {labels_dir}")

    missing_images = 0
    unreadable_images = 0

    for label_path in label_files:
        stem = label_path.stem
        if stem not in img_map:
            missing_images += 1
            continue

        img_path = img_map[stem]
        img = cv2.imread(str(img_path))
        if img is None:
            unreadable_images += 1
            continue
        H, W = img.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            bw = float(parts[3]) * W
            bh = float(parts[4]) * H

            sum_w[cls] = sum_w.get(cls, 0.0) + bw
            sum_h[cls] = sum_h.get(cls, 0.0) + bh
            count[cls] = count.get(cls, 0) + 1

    classes = sorted(count.keys())
    avg_w = np.array([sum_w[c] / count[c] for c in classes], dtype=float)
    avg_h = np.array([sum_h[c] / count[c] for c in classes], dtype=float)
    n = np.array([count[c] for c in classes], dtype=int)

    if class_names is None:
        labels = [str(c) for c in classes]
    else:
        labels = [class_names[c] if c < len(class_names) else str(c) for c in classes]

    if missing_images:
        print(f"[warn] {missing_images} label files had no matching image file stem in {images_dir}", file=sys.stderr)
    if unreadable_images:
        print(f"[warn] {unreadable_images} matching images could not be read (cv2.imread failed)", file=sys.stderr)

    return classes, labels, avg_w, avg_h, n


def get_per_class_ap_ultralytics_rtdetr(weights_path: str, data_yaml: str, split="val", imgsz=640):
    """
    Runs Ultralytics validation and returns per-class AP (usually mAP@0.5:0.95 per class).
    Tries multiple attribute names to be robust across Ultralytics versions.

    Returns:
      ap_per_class: np.ndarray shape [num_classes] OR [num_classes, ...]
    """
    try:
        from ultralytics import RTDETR
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is required for automatic per-class AP extraction.\n"
            "Install with: pip install ultralytics\n"
            f"Import error: {e}"
        )

    model = RTDETR(weights_path)
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz)

    # Common patterns by version:
    # metrics.box.maps  -> per-class mAP@0.5:0.95
    # metrics.box.ap    -> sometimes per-class AP
    # metrics.box.ap50  -> per-class AP@0.5
    candidates = ["maps", "ap", "ap50"]

    per_class = None
    for attr in candidates:
        if hasattr(metrics.box, attr):
            v = getattr(metrics.box, attr)
            try:
                per_class = np.array(v, dtype=float)
                break
            except Exception:
                pass

    if per_class is None:
        # Fallback: show what exists for debugging
        available = [a for a in dir(metrics.box) if not a.startswith("_")]
        raise RuntimeError(
            "Could not locate per-class AP in Ultralytics metrics.\n"
            f"Available metrics.box attrs: {available}\n"
            "Tip: print(metrics) / print(dir(metrics.box)) in your environment and adjust."
        )

    return per_class


def plot_avg_bbox_vs_ap(
    labels,
    avg_w,
    avg_h,
    ap,
    counts,
    title="Average bbox sizes by category and respective AP",
    ap_scale=1.0,
    bubble_scale=8.0,   # kept for compatibility; not used anymore
    cmap="coolwarm",
    out_path=None,
    show=True,
):
    """
    Bubble plot:
      x = avg_w (px)
      y = avg_h (px)
      size = AP-scaled (matches original script)
      color = AP
    """
    ap = np.array(ap, dtype=float) * float(ap_scale)
    avg_w = np.array(avg_w, dtype=float)
    avg_h = np.array(avg_h, dtype=float)

    # ---- AP-based bubble sizing (ORIGINAL LOGIC) ----
    ap_scaled = ap.copy()
    max_ap = np.nanmax(ap_scaled)
    if max_ap > 0:
        sizes = 1000.0 * (ap_scaled / max_ap)
    else:
        sizes = np.full_like(ap_scaled, 100.0)

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(
        avg_w,
        avg_h,
        s=sizes,
        c=ap,
        cmap=cmap,
        edgecolors="k",
        linewidths=1,
        alpha=0.9,
    )

    ax.set_title(title)
    ax.set_xlabel("Average BBox Width (px)")
    ax.set_ylabel("Average BBox Height (px)")
    ax.grid(True, alpha=0.25)

    for x, y, name in zip(avg_w, avg_h, labels):
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
            weight="bold",
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Average Precision (AP)")

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"[ok] saved plot to: {out_path}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained RT-DETR weights (Ultralytics .pt).")
    ap.add_argument("--data", required=True, help="Path to Ultralytics-style data.yaml.")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to use.")
    ap.add_argument("--imgsz", type=int, default=1024, help="Validation image size for Ultralytics val().")
    ap.add_argument("--images", default=None, help="Override images directory for the split.")
    ap.add_argument("--labels", default=None, help="Override labels directory for the split.")
    ap.add_argument("--ap-scale", type=float, default=100.0, help="Multiply AP by this (100 -> percent).")
    ap.add_argument("--bubble-scale", type=float, default=1.0, help="Bubble size scaling factor.")
    ap.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap name.")
    ap.add_argument("--out", default="bbox_ap_plot.png", help="Output plot filename (png).")
    ap.add_argument("--no-show", action="store_true", help="Do not open an interactive window.")
    args = ap.parse_args()

    data_yaml = _as_path(args.data)
    weights = _as_path(args.weights)

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    with open(data_yaml, "r", encoding="utf-8") as f:
        data_dict = yaml.safe_load(f)

    class_names = data_dict.get("names", None)

    # Determine images/labels dirs
    if args.images is not None:
        images_dir = args.images
    else:
        images_dir = _resolve_split_path(data_dict, args.split)

    if args.labels is not None:
        labels_dir = args.labels
    else:
        labels_dir = _default_labels_dir_from_images(images_dir)

    print(f"[info] images_dir: {images_dir}")
    print(f"[info] labels_dir: {labels_dir}")

    # 1) GT stats
    classes, labels, avg_w, avg_h, counts = load_yolo_gt_stats(
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_names=class_names,
    )
    print(f"[info] found {len(classes)} classes with GT boxes in '{args.split}' split")

    # 2) Per-class AP from RT-DETR
    ap_all = get_per_class_ap_ultralytics_rtdetr(
        weights_path=str(weights),
        data_yaml=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
    )

    # Handle AP shapes:
    # - If ap_all is 1D: [C]
    # - If ap_all is 2D: [C, K] (e.g., per-IoU buckets) -> mean over axis=1
    ap_all = np.array(ap_all, dtype=float)
    if ap_all.ndim == 2:
        ap_all = ap_all.mean(axis=1)

    # Align AP to classes observed in GT
    ap_per_class = np.array(
        [ap_all[c] if c < len(ap_all) else np.nan for c in classes],
        dtype=float,
    )

    # 3) Plot
    plot_avg_bbox_vs_ap(
        labels=labels,
        avg_w=avg_w,
        avg_h=avg_h,
        ap=ap_per_class,
        counts=counts,
        title="Average bbox sizes by category and respective AP",
        ap_scale=args.ap_scale,
        bubble_scale=args.bubble_scale,
        cmap=args.cmap,
        out_path=args.out,
        show=(not args.no_show),
    )


if __name__ == "__main__":
    main()
