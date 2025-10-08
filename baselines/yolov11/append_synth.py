#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append g synthetic YOLO samples per class to an existing few-shot YOLO dataset.

Handles nested folders (e.g. images/train/batch_3/…).
"""

import os, random, argparse, shutil
from glob import glob
from collections import defaultdict

# -------------------------
# Helpers
# -------------------------
def parse_yolo_label(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                boxes.append(cls)
    return boxes


def count_instances(label_dir):
    counts = defaultdict(int)
    for lbl in glob(os.path.join(label_dir, "**", "*.txt"), recursive=True):
        for cls in parse_yolo_label(lbl):
            counts[cls] += 1
    return counts


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
# Main merge logic
# -------------------------
def add_g_synthetic(fewshot_dir, synth_dir, out_dir, g_per_class=20, seed=0):
    random.seed(seed)
    few_train_img = os.path.join(fewshot_dir, "images/train")
    few_train_lbl = os.path.join(fewshot_dir, "labels/train")
    synth_img = os.path.join(synth_dir, "images")
    synth_lbl = os.path.join(synth_dir, "labels")

    out_img = os.path.join(out_dir, "images/train")
    out_lbl = os.path.join(out_dir, "labels/train")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    # ---- Copy existing few-shot train ----
    print("📂 Copying existing few-shot train set...")
    copy_tree(few_train_img, out_img)
    copy_tree(few_train_lbl, out_lbl)

    # ---- Prepare synthetic grouping ----
    print("🔍 Indexing synthetic labels...")
    synth_labels = glob(os.path.join(synth_lbl, "**", "*.txt"), recursive=True)
    synth_by_class = defaultdict(list)
    for lbl in synth_labels:
        cls_ids = parse_yolo_label(lbl)
        if not cls_ids:
            continue
        for cls in set(cls_ids):
            synth_by_class[cls].append(lbl)

    # ---- Sample g label files per class ----
    selected = set()
    for cls, files in synth_by_class.items():
        random.shuffle(files)
        chosen = files[:g_per_class]
        for f in chosen:
            selected.add(f)

    print(f"✅ Selected {len(selected)} synthetic label files (~{g_per_class} per class).")

    # ---- Copy selected synthetic samples ----
    for lbl_path in selected:
        # copy label
        dst_lbl_path = os.path.join(out_lbl, os.path.basename(lbl_path))
        os.makedirs(os.path.dirname(dst_lbl_path), exist_ok=True)
        shutil.copy2(lbl_path, dst_lbl_path)

        # find matching image
        name = os.path.splitext(os.path.basename(lbl_path))[0]
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(synth_img, name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path:
            # search recursively
            found = glob(os.path.join(synth_img, "**", name + ".*"), recursive=True)
            if found:
                img_path = found[0]
        if img_path:
            shutil.copy2(img_path, os.path.join(out_img, os.path.basename(img_path)))

    # ---- Copy val/test unchanged ----
    for split in ["val", "test"]:
        for sub in ["images", "labels"]:
            src = os.path.join(fewshot_dir, sub, split)
            dst = os.path.join(out_dir, sub, split)
            print(f"📂 Copying {split} {sub}...")
            copy_tree(src, dst)

    print("\n✅ Done!")
    print(f"New dataset saved to: {out_dir}")
    print(f"  images/train → {out_img}")
    print(f"  labels/train → {out_lbl}")


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Append g synthetic YOLO samples per class to few-shot train split.")
    ap.add_argument("--fewshot-dir", required=True, help="Path to YOLO few-shot dataset root")
    ap.add_argument("--synthetic-dir", required=True, help="Path to YOLO synthetic dataset root")
    ap.add_argument("--out-dir", required=True, help="Output merged dataset path")
    ap.add_argument("--g", type=int, default=20, help="Synthetic instances per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    add_g_synthetic(args.fewshot_dir, args.synthetic_dir, args.out_dir, g_per_class=args.g, seed=args.seed)


if __name__ == "__main__":
    main()
