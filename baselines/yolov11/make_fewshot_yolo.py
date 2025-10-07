#!/usr/bin/env python3
"""
Few-shot YOLO dataset from COCO, using SUPER-CATEGORIES as classes.

Features
- --classes takes SUPER-CATEGORY names (order preserved for IDs 0..N-1)
- K-shot by INSTANCES per supercategory for TRAIN (not by images)
- Remaining images -> VAL/TEST split
- Relabel leaf categories to their supercategory IDs
- Preserves subfolders from images_root (e.g., batch1/, batch2/) in BOTH images and labels
- Outputs:
    out/
      images/{train,val,test}/<subdirs>/image.jpg
      labels/{train,val,test}/<subdirs>/image.txt
      dataset.yaml          # with quoted class names
      supercat_mapping.json # diagnostics
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# ----------------------------- utils -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def place_image(src: Path, dst: Path, mode: str = "symlink") -> None:
    ensure_dir(dst.parent)
    if mode == "symlink":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            target = Path(os.path.relpath(src, start=dst.parent))
            dst.symlink_to(target)
        except Exception:
            shutil.copy2(src, dst)  # fallback if symlink not allowed
    elif mode == "copy":
        if not dst.exists():
            shutil.copy2(src, dst)
    elif mode == "move":
        if not dst.exists():
            shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_coco(p: Path) -> dict:
    coco = json.load(open(p, "r"))
    assert {"images", "annotations", "categories"} <= set(coco), "Not a COCO detection JSON."
    return coco


def build_id2cat(categories):
    # id -> {"name": ..., "super": ...}
    return {c["id"]: {"name": c["name"], "super": c.get("supercategory", "")} for c in categories}


def resolve_supercats(categories, classes_arg: str):
    """
    Returns:
      super_names: list[str]             -> ordered per user (if provided) else all supercats sorted by name
      super2new:   dict[str,int]         -> supercat name -> new contiguous id (0..N-1)
    """
    all_supers = {c.get("supercategory", "") for c in categories}
    if classes_arg:
        given = [s.strip() for s in classes_arg.split(",") if s.strip()]
        for s in given:
            if s not in all_supers:
                raise ValueError(f"Supercategory '{s}' not found in dataset.")
        super_names = given  # preserve user order
    else:
        super_names = sorted(all_supers)
    super2new = {s: i for i, s in enumerate(super_names)}
    return super_names, super2new


def rel_from_images_root(file_name: str, images_root: Path) -> Path:
    """
    Return path of this image RELATIVE to images_root, preserving any subfolders (batch1/, batch2/, ...).
    Tries multiple heuristics so it works whether file_name is 'batch1/x.jpg', 'images/train/batch1/x.jpg', or just 'x.jpg'.
    """
    fn = file_name.lstrip("/")
    cand = images_root / fn
    if cand.exists():
        try:
            return cand.relative_to(images_root)
        except Exception:
            pass

    # try basename only
    base = Path(fn).name
    for p in images_root.rglob(base):
        if p.is_file():
            try:
                return p.relative_to(images_root)
            except Exception:
                return Path(base)

    # fallback: use provided string as relative
    return Path(fn)


# ------------------------ selection / splitting ------------------------

def pick_k_per_super(coco, id2cat, super2new, K: int, seed: int = 42):
    """
    Select exactly K annotations per supercategory (if fewer exist, take all).
    Return:
      train_img_ids (set[int]),
      train_ann_ids_per_img (dict[img_id] -> list[ann_id]),
      remaining_img_ids (list[int]),
      per_super_available (dict[super] -> int)
    """
    random.seed(seed)

    anns_by_super = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) == 1 or "bbox" not in a:
            continue
        sup = id2cat[a["category_id"]]["super"]
        if sup in super2new:
            anns_by_super[sup].append(a)

    per_super_available = {s: len(anns_by_super[s]) for s in super2new}

    selected_ann_ids = set()
    for s, lst in anns_by_super.items():
        random.shuffle(lst)
        take = lst[:K] if len(lst) >= K else lst[:]  # take all if fewer than K
        selected_ann_ids.update(a["id"] for a in take)

    train_img_ids = set()
    train_ann_ids_per_img = defaultdict(list)
    for a in coco["annotations"]:
        if a["id"] in selected_ann_ids:
            train_img_ids.add(a["image_id"])
            train_ann_ids_per_img[a["image_id"]].append(a["id"])

    all_img_ids = {im["id"] for im in coco["images"]}
    remaining_img_ids = sorted(list(all_img_ids - train_img_ids))
    return train_img_ids, train_ann_ids_per_img, remaining_img_ids, per_super_available


def split_remaining(remaining_img_ids, val_ratio=0.5, seed=42):
    random.seed(seed)
    ids = remaining_img_ids[:]
    random.shuffle(ids)
    n = len(ids)
    n_val = int(val_ratio * n)
    return set(ids[:n_val]), set(ids[n_val:])


# ------------------------- label conversion ----------------------------

def coco_to_yolo_line(ann, W, H, id2cat, super2new):
    x, y, w, h = ann["bbox"]
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    ww = w / W
    hh = h / H
    # clamp
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    ww = max(0.0, min(1.0, ww))
    hh = max(0.0, min(1.0, hh))
    sup = id2cat[ann["category_id"]]["super"]
    cls = super2new[sup]
    return f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}"


def write_split_yolo(
    coco,
    img_ids: set,
    ann_filter_per_img: dict | None,      # if None -> keep all allowed supers; else keep only selected ann IDs
    out_images_dir: Path,
    out_labels_dir: Path,
    images_root: Path,
    id2cat: dict,
    super2new: dict,
    allowed_supers: set | None,
    mode: str = "symlink",
):
    imgs_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    wrote_i = wrote_l = 0
    for img_id in img_ids:
        im = imgs_by_id[img_id]
        rel_src = rel_from_images_root(im["file_name"], images_root)  # preserve subfolders
        src = images_root / rel_src
        if not src.exists():
            print(f"[WARN] missing image: {im['file_name']} (resolved {src})", file=sys.stderr)
            continue

        dst_img = out_images_dir / rel_src
        place_image(src, dst_img, mode=mode)
        wrote_i += 1

        # Labels mirror the same subfolder structure
        label_path = (out_labels_dir / rel_src).with_suffix(".txt")
        ensure_dir(label_path.parent)

        W, H = im.get("width"), im.get("height")
        lines = []
        if ann_filter_per_img is None:
            # VAL/TEST: keep all anns whose super in allowed_supers
            for a in anns_by_img[img_id]:
                if a.get("iscrowd", 0) == 1 or "bbox" not in a:
                    continue
                sup = id2cat[a["category_id"]]["super"]
                if allowed_supers and sup not in allowed_supers:
                    continue
                lines.append(coco_to_yolo_line(a, W, H, id2cat, super2new))
        else:
            # TRAIN: keep only selected instance IDs for this image
            keep_ids = set(ann_filter_per_img.get(img_id, []))
            for a in anns_by_img[img_id]:
                if a["id"] not in keep_ids:
                    continue
                if a.get("iscrowd", 0) == 1 or "bbox" not in a:
                    continue
                lines.append(coco_to_yolo_line(a, W, H, id2cat, super2new))

        label_path.write_text("\n".join(lines))
        wrote_l += 1

    return wrote_i, wrote_l


# ----------------------------- main ------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build a K-shot YOLO dataset from COCO using SUPER-CATEGORIES as classes (order preserved)."
    )
    ap.add_argument("--ann", required=True, help="Path to COCO annotations JSON (real dataset)")
    ap.add_argument("--images-root", required=True, help="Root folder containing the images")
    ap.add_argument("--out", required=True, help="Output dataset root (YOLO layout will be created)")
    ap.add_argument("--k", type=int, required=True, help="K instances per supercategory for TRAIN")
    ap.add_argument("--classes", type=str, default="", help="Comma-separated SUPER-CATEGORY names (order preserved). Default: all")
    ap.add_argument("--val-ratio", type=float, default=0.5, help="Fraction of remaining images to VAL (rest TEST)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["symlink", "copy", "move"], default="symlink", help="How to place images")
    args = ap.parse_args()

    ann_path = Path(args.ann)
    images_root = Path(args.images_root)
    out_root = Path(args.out)

    if not ann_path.exists():
        print(f"[ERR] annotations not found: {ann_path}", file=sys.stderr); sys.exit(1)
    if not images_root.exists():
        print(f"[ERR] images_root not found: {images_root}", file=sys.stderr); sys.exit(1)

    coco = load_coco(ann_path)
    id2cat = build_id2cat(coco["categories"])
    super_names, super2new = resolve_supercats(coco["categories"], args.classes)
    allowed_supers = set(super_names)

    # Pick K instances per supercategory for TRAIN
    train_img_ids, train_ann_ids_per_img, remaining_img_ids, per_super_avail = pick_k_per_super(
        coco, id2cat, super2new, args.k, seed=args.seed
    )

    # Split remaining into VAL / TEST
    val_ids, test_ids = split_remaining(remaining_img_ids, val_ratio=args.val_ratio, seed=args.seed)

    # Create folders
    imgs_tr = out_root / "images" / "train"
    imgs_va = out_root / "images" / "val"
    imgs_te = out_root / "images" / "test"
    lbls_tr = out_root / "labels" / "train"
    lbls_va = out_root / "labels" / "val"
    lbls_te = out_root / "labels" / "test"
    for d in [imgs_tr, imgs_va, imgs_te, lbls_tr, lbls_va, lbls_te]:
        ensure_dir(d)

    # Write splits (train keeps only selected instance IDs; val/test keep all anns of chosen supers)
    wi_tr, wl_tr = write_split_yolo(
        coco, train_img_ids, train_ann_ids_per_img, imgs_tr, lbls_tr,
        images_root, id2cat, super2new, allowed_supers=None, mode=args.mode
    )
    wi_va, wl_va = write_split_yolo(
        coco, val_ids, None, imgs_va, lbls_va,
        images_root, id2cat, super2new, allowed_supers=allowed_supers, mode=args.mode
    )
    wi_te, wl_te = write_split_yolo(
        coco, test_ids, None, imgs_te, lbls_te,
        images_root, id2cat, super2new, allowed_supers=allowed_supers, mode=args.mode
    )

    # Write dataset.yaml (quote names for safety)
    yml = [
        f'path: {out_root}',
        'train: images/train',
        'val: images/val',
        'test: images/test',
        'names:'
    ] + [f'  {i}: "{n}"' for i, n in enumerate(super_names)]
    (out_root / "dataset.yaml").write_text("\n".join(yml))

    # Diagnostics
    mapping = {
        "super_names_in_order": super_names,
        "super2new": super2new,
        "per_super_available_instances": per_super_avail
    }
    (out_root / "supercat_mapping.json").write_text(json.dumps(mapping, indent=2))

    # Report
    print("\n=== Few-shot by SUPER-CATEGORY ===")
    print("Classes (supercategories, in order):", ", ".join(super_names))
    print(f"K per supercategory (train): {args.k}")
    for s in super_names:
        picked = sum(
            1 for ids in train_ann_ids_per_img.values() for aid in ids
            if id2cat[next(a["category_id"] for a in coco["annotations"] if a["id"] == aid)]["super"] == s
        )
        print(f"  - {s}: picked {picked} / available {per_super_avail.get(s, 0)}")
    print(f"Train images: {len(train_img_ids)}  (labels: {wl_tr})")
    print(f"Val   images: {len(val_ids)}       (labels: {wl_va})")
    print(f"Test  images: {len(test_ids)}      (labels: {wl_te})")
    print("\nDataset YAML:", out_root / "dataset.yaml")
    print("Mapping JSON:", out_root / "supercat_mapping.json")

if __name__ == "__main__":
    main()
