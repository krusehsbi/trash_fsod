#!/usr/bin/env python3
"""
Split a COCO dataset into train/val[/test] and emit **COCO JSONs** and **copied images per split**.
Allows replacing category names using a provided list.

Usage examples:
  python split_coco_to_coco.py \
      --images-dir /data/mycoco/images \
      --coco-json /data/mycoco/annotations.json \
      --out-root /data/mycoco/splits \
      --splits 80 20 \
      --seed 1337 \
      --category-names /path/to/classes.txt

Notes:
- Deterministic split per --seed.
- Keeps the original COCO category IDs; replaces their 'name' with entries from --category-names if given.
- Use --prune-unused-categories to drop categories not present in each split.
- Copies the corresponding images into images/{split}/ directories.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def set_seed(seed: int):
    random.seed(seed)

def load_coco(json_path: Path) -> Dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ["images", "annotations", "categories"]:
        if key not in data:
            raise ValueError(f"COCO file missing '{key}'")
    for im in data["images"]:
        im["id"] = int(im["id"])
        im["width"] = int(im["width"])
        im["height"] = int(im["height"])
    for ann in data["annotations"]:
        ann["id"] = int(ann.get("id", 0))
        ann["image_id"] = int(ann["image_id"])
        ann["category_id"] = int(ann["category_id"])
    for cat in data["categories"]:
        cat["id"] = int(cat["id"])
    return data

def apply_category_names(data: Dict, names_path: Path):
    if not names_path.exists():
        raise FileNotFoundError(f"Category names file not found: {names_path}")
    names = [ln.strip() for ln in names_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cats = data["categories"]
    for i, cat in enumerate(sorted(cats, key=lambda c: c["id"])):
        if i < len(names):
            cat["name"] = names[i]
    data["categories"] = cats

def split_images(image_ids: List[int], splits: List[int]) -> List[List[int]]:
    assert sum(splits) == 100, "--splits must sum to 100"
    ids = image_ids.copy()
    random.shuffle(ids)
    n = len(ids)
    parts = []
    start = 0
    for i, pct in enumerate(splits):
        k = round(n * pct / 100.0)
        end = n if i == len(splits) - 1 else min(n, start + k)
        parts.append(ids[start:end])
        start = end
    return parts

def coco_subset(data: Dict, keep_img_ids: Set[int], prune_unused_categories: bool) -> Dict:
    images = [im for im in data["images"] if im["id"] in keep_img_ids]
    img_id_set = {im["id"] for im in images}
    anns = [ann for ann in data["annotations"] if ann["image_id"] in img_id_set]

    if prune_unused_categories:
        used_cat_ids = {ann["category_id"] for ann in anns}
        categories = [c for c in data["categories"] if c["id"] in used_cat_ids]
    else:
        categories = data["categories"]

    return {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": categories,
    }

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def write_coco_json(out_path: Path, coco_obj: Dict, pretty: bool = False):
    ensure_parent(out_path)
    if pretty:
        out_path.write_text(json.dumps(coco_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        out_path.write_text(json.dumps(coco_obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")

def guess_image_path(images_dir: Path, file_name: str) -> Path:
    cand = images_dir / file_name
    if cand.exists():
        return cand
    base = Path(file_name).name
    cand2 = images_dir / base
    if cand2.exists():
        return cand2
    stem = Path(base).stem
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.stem == stem:
            return p
    return cand

def copy_images(split_name: str, split_images: List[Dict], images_dir: Path, out_root: Path):
    img_out_dir = out_root / "images" / split_name
    for im in split_images:
        rel = Path(im["file_name"])
        src = guess_image_path(images_dir, im["file_name"])
        dst = img_out_dir / rel
        ensure_parent(dst)
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser(description="Split COCO into COCO-per-split JSONs and copy corresponding images.")
    ap.add_argument("--images-dir", required=True, type=str, help="Path to images root directory")
    ap.add_argument("--coco-json", required=True, type=str, help="Path to COCO instances JSON")
    ap.add_argument("--out-root", default="splits", type=str, help="Output root directory")
    ap.add_argument("--splits", nargs="+", type=int, default=[80, 20], help="Percentages that sum to 100")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for deterministic shuffling")
    ap.add_argument("--prune-unused-categories", action="store_true", help="Drop unused categories")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    ap.add_argument("--split-names", nargs="+", default=None)
    ap.add_argument("--category-names", type=str, default=None, help="Path to text file with new category names")

    args = ap.parse_args()

    set_seed(args.seed)

    coco_json = Path(args.coco_json).resolve()
    out_root = Path(args.out_root).resolve()
    images_dir = Path(args.images_dir).resolve()

    data = load_coco(coco_json)

    if args.category_names:
        apply_category_names(data, Path(args.category_names))

    num_parts = len(args.splits)
    split_names = args.split_names if args.split_names else ["train", "val", "test"][:num_parts]
    if len(split_names) != num_parts:
        raise ValueError("--split-names length must match number of splits")
    if sum(args.splits) != 100:
        raise ValueError("--splits must sum to 100")

    image_ids = [im["id"] for im in data["images"]]
    id_parts = split_images(image_ids, args.splits)

    for split_name, ids in zip(split_names, id_parts):
        subset = coco_subset(data, set(ids), prune_unused_categories=args.prune_unused_categories)
        ann_dir = out_root / "annotations"
        write_coco_json(ann_dir / f"instances_{split_name}.json", subset, pretty=args.pretty)
        copy_images(split_name, subset["images"], images_dir, out_root)

    print("[OK] COCO→COCO split with copied images done. Seed:", args.seed)

if __name__ == "__main__":
    main()
