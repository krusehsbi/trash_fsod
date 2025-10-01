#!/usr/bin/env python3
import argparse, json, random, shutil, sys, os
from pathlib import Path

SPLITS = {"train", "val", "test"}
IMAGE_DIR_TOKENS = {"images", "image", "img"}

def resolve_image_path(img_record, images_root: Path) -> Path:
    fp = img_record.get("file_name", "")
    p = Path(fp)
    if p.is_absolute():
        return p
    cand = images_root / fp
    if cand.exists():
        return cand
    return images_root / p.name

def normalized_rel_subpath(img_record, images_root: Path) -> Path:
    """
    Compute a clean per-image subpath (without leading images/ or split/),
    so we can place it under images/{split}/<subpath>.
    """
    src = resolve_image_path(img_record, images_root)
    try:
        rel = src.relative_to(images_root)
    except Exception:
        rel = Path(img_record.get("file_name", Path(src).name))

    parts = list(rel.parts)
    while parts and parts[0].lower() in IMAGE_DIR_TOKENS:
        parts = parts[1:]
    if parts and parts[0].lower() in SPLITS:
        parts = parts[1:]
    if not parts:
        parts = [rel.name]
    return Path(*parts)

def materialize_split_files(coco, image_ids_set, images_root: Path, out_images_split_dir: Path, mode: str):
    out_images_split_dir.mkdir(parents=True, exist_ok=True)
    ok = missing = 0
    for img in coco["images"]:
        if img["id"] not in image_ids_set:
            continue
        src = resolve_image_path(img, images_root)
        if not src.exists():
            missing += 1
            print(f"[WARN] Missing image for id={img['id']}: expected {src}", file=sys.stderr)
            continue
        rel = normalized_rel_subpath(img, images_root)
        dst = out_images_split_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if mode == "symlink":
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                target = Path(os.path.relpath(src, start=dst.parent))
                dst.symlink_to(target)
            elif mode == "copy":
                if not dst.exists():
                    shutil.copy2(src, dst)
            elif mode == "move":
                if not dst.exists():
                    shutil.move(str(src), str(dst))
            else:
                raise ValueError(f"Unknown mode: {mode}")
            ok += 1
        except Exception as e:
            print(f"[ERROR] Failed to place {src} -> {dst}: {e}", file=sys.stderr)
    return ok, missing

def split_ids_uniform(images, splits, seed):
    random.seed(seed)
    ids = [img["id"] for img in images]
    random.shuffle(ids)
    n = len(ids)
    n_train = int(splits[0] * n)
    n_val = int(splits[1] * n)
    return set(ids[:n_train]), set(ids[n_train:n_train + n_val]), set(ids[n_train + n_val:])

def filter_out_missing_images(coco, images_root: Path):
    kept_images, kept_ids, dropped = [], set(), 0
    for img in coco["images"]:
        if resolve_image_path(img, images_root).exists():
            kept_images.append(img); kept_ids.add(img["id"])
        else:
            dropped += 1
    anns = [a for a in coco["annotations"] if a["image_id"] in kept_ids]
    return {"images": kept_images, "annotations": anns, "categories": coco["categories"]}, dropped

def make_cat_mapping(categories):
    # Stable mapping: categories sorted by original id -> 0..N-1
    cats_sorted = sorted(categories, key=lambda c: c["id"])
    old2new = {c["id"]: i for i, c in enumerate(cats_sorted)}
    names = [c["name"] for c in cats_sorted]
    return old2new, names

def write_yolo_labels(coco, image_ids_set, labels_root: Path, split_name: str, idmap):
    labels_dir = labels_root / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)
    imgs = {im["id"]: im for im in coco["images"] if im["id"] in image_ids_set}
    per_img = {}
    for ann in coco["annotations"]:
        if ann.get("image_id") in image_ids_set:
            per_img.setdefault(ann["image_id"], []).append(ann)

    wrote = 0
    for img_id, im in imgs.items():
        w, h = im["width"], im["height"]
        # derive label filename from (normalized) image relative path
        # ensure subpath matches what materialize_split_files used
        subpath = normalized_rel_subpath(im, Path("."))  # images_root not needed for naming
        stem = Path(subpath).with_suffix(".txt").name
        out_path = labels_dir / stem

        lines = []
        for ann in per_img.get(img_id, []):
            if ann.get("iscrowd", 0) == 1:  # skip crowd
                continue
            if "bbox" not in ann:
                continue
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            ww = bw / w
            hh = bh / h
            cls = idmap[ann["category_id"]]
            # clamp just in case
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            ww = max(0.0, min(1.0, ww))
            hh = max(0.0, min(1.0, hh))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

        out_path.write_text("\n".join(lines))
        wrote += 1
    return wrote

def write_dataset_yaml(out_dir: Path, names):
    # YOLO data YAML pointing to images dirs; labels are discovered automatically under labels/
    yml = [
        f'path: {out_dir}',
        'train: images/train',
        'val: images/val',
        'test: images/test',
        'names:'
    ]
    for i, n in enumerate(names):
        yml.append(f'  {i}: {n}')
    (out_dir / "dataset.yaml").write_text("\n".join(yml))

def main():
    parser = argparse.ArgumentParser(
        description="Split COCO JSON and build YOLO-format dataset (images/ + labels/ wrappers)."
    )
    parser.add_argument("json_path", type=str, help="Path to input COCO annotations JSON")
    parser.add_argument("images_root", type=str, help="Root directory that contains the images")
    parser.add_argument("out_dir", type=str, help="Output dataset root")
    parser.add_argument("--splits", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "symlink", "move"], default="symlink")
    parser.add_argument("--prefilter-missing", action="store_true")
    args = parser.parse_args()

    json_path, images_root, out_dir = Path(args.json_path), Path(args.images_root), Path(args.out_dir)
    if not json_path.exists():
        print(f"[ERR] JSON not found: {json_path}", file=sys.stderr); sys.exit(1)
    if not images_root.exists():
        print(f"[ERR] images_root not found: {images_root}", file=sys.stderr); sys.exit(1)

    coco = json.load(open(json_path))
    if args.prefilter_missing:
        coco, dropped = filter_out_missing_images(coco, images_root)
        if dropped:
            print(f"[INFO] Prefilter: dropped {dropped} images not found on disk.")

    train_ids, val_ids, test_ids = split_ids_uniform(coco["images"], args.splits, args.seed)
    idmap, names = make_cat_mapping(coco["categories"])

    # Make folders (YOLO standard)
    imgs_train = out_dir / "images" / "train"
    imgs_val   = out_dir / "images" / "val"
    imgs_test  = out_dir / "images" / "test"
    labels_root = out_dir / "labels"
    for d in (imgs_train, imgs_val, imgs_test, labels_root):
        d.mkdir(parents=True, exist_ok=True)

    # Place images per split
    ok_tr, miss_tr = materialize_split_files(coco, train_ids, images_root, imgs_train, args.mode)
    ok_va, miss_va = materialize_split_files(coco, val_ids,   images_root, imgs_val,   args.mode)
    ok_te, miss_te = materialize_split_files(coco, test_ids,  images_root, imgs_test,  args.mode)

    # Write YOLO labels per split
    wrote_tr = write_yolo_labels(coco, train_ids, labels_root, "train", idmap)
    wrote_va = write_yolo_labels(coco, val_ids,   labels_root, "val",   idmap)
    wrote_te = write_yolo_labels(coco, test_ids,  labels_root, "test",  idmap)

    # Write dataset.yaml (names included)
    write_dataset_yaml(out_dir, names)

    print("\n=== Summary ===")
    print(f"Train: images placed={ok_tr} (missing={miss_tr}), labels written for {wrote_tr} images")
    print(f"Val  : images placed={ok_va} (missing={miss_va}), labels written for {wrote_va} images")
    print(f"Test : images placed={ok_te} (missing={miss_te}), labels written for {wrote_te} images")
    print(f"\nDataset root: {out_dir}")
    print("Use this data YAML:")
    print(f"  data: {out_dir/'dataset.yaml'}")
    print("Then run YOLO with that data YAML.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse, json, random, shutil, sys, os
from pathlib import Path

SPLITS = {"train", "val", "test"}
IMAGE_DIR_TOKENS = {"images", "image", "img"}

def resolve_image_path(img_record, images_root: Path) -> Path:
    fp = img_record.get("file_name", "")
    p = Path(fp)
    if p.is_absolute():
        return p
    cand = images_root / fp
    if cand.exists():
        return cand
    return images_root / p.name

def normalized_rel_subpath(img_record, images_root: Path) -> Path:
    """
    Compute a clean per-image subpath (without leading images/ or split/),
    so we can place it under images/{split}/<subpath>.
    """
    src = resolve_image_path(img_record, images_root)
    try:
        rel = src.relative_to(images_root)
    except Exception:
        rel = Path(img_record.get("file_name", Path(src).name))

    parts = list(rel.parts)
    while parts and parts[0].lower() in IMAGE_DIR_TOKENS:
        parts = parts[1:]
    if parts and parts[0].lower() in SPLITS:
        parts = parts[1:]
    if not parts:
        parts = [rel.name]
    return Path(*parts)

def materialize_split_files(coco, image_ids_set, images_root: Path, out_images_split_dir: Path, mode: str):
    out_images_split_dir.mkdir(parents=True, exist_ok=True)
    ok = missing = 0
    for img in coco["images"]:
        if img["id"] not in image_ids_set:
            continue
        src = resolve_image_path(img, images_root)
        if not src.exists():
            missing += 1
            print(f"[WARN] Missing image for id={img['id']}: expected {src}", file=sys.stderr)
            continue
        rel = normalized_rel_subpath(img, images_root)
        dst = out_images_split_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if mode == "symlink":
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                target = Path(os.path.relpath(src, start=dst.parent))
                dst.symlink_to(target)
            elif mode == "copy":
                if not dst.exists():
                    shutil.copy2(src, dst)
            elif mode == "move":
                if not dst.exists():
                    shutil.move(str(src), str(dst))
            else:
                raise ValueError(f"Unknown mode: {mode}")
            ok += 1
        except Exception as e:
            print(f"[ERROR] Failed to place {src} -> {dst}: {e}", file=sys.stderr)
    return ok, missing

def split_ids_uniform(images, splits, seed):
    random.seed(seed)
    ids = [img["id"] for img in images]
    random.shuffle(ids)
    n = len(ids)
    n_train = int(splits[0] * n)
    n_val = int(splits[1] * n)
    return set(ids[:n_train]), set(ids[n_train:n_train + n_val]), set(ids[n_train + n_val:])

def filter_out_missing_images(coco, images_root: Path):
    kept_images, kept_ids, dropped = [], set(), 0
    for img in coco["images"]:
        if resolve_image_path(img, images_root).exists():
            kept_images.append(img); kept_ids.add(img["id"])
        else:
            dropped += 1
    anns = [a for a in coco["annotations"] if a["image_id"] in kept_ids]
    return {"images": kept_images, "annotations": anns, "categories": coco["categories"]}, dropped

def make_cat_mapping(categories):
    # Stable mapping: categories sorted by original id -> 0..N-1
    cats_sorted = sorted(categories, key=lambda c: c["id"])
    old2new = {c["id"]: i for i, c in enumerate(cats_sorted)}
    names = [c["name"] for c in cats_sorted]
    return old2new, names

def write_yolo_labels(coco, image_ids_set, labels_root: Path, split_name: str, idmap):
    labels_dir = labels_root / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)
    imgs = {im["id"]: im for im in coco["images"] if im["id"] in image_ids_set}
    per_img = {}
    for ann in coco["annotations"]:
        if ann.get("image_id") in image_ids_set:
            per_img.setdefault(ann["image_id"], []).append(ann)

    wrote = 0
    for img_id, im in imgs.items():
        w, h = im["width"], im["height"]
        # derive label filename from (normalized) image relative path
        # ensure subpath matches what materialize_split_files used
        subpath = normalized_rel_subpath(im, Path("."))  # images_root not needed for naming
        stem = Path(subpath).with_suffix(".txt").name
        out_path = labels_dir / stem

        lines = []
        for ann in per_img.get(img_id, []):
            if ann.get("iscrowd", 0) == 1:  # skip crowd
                continue
            if "bbox" not in ann:
                continue
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            ww = bw / w
            hh = bh / h
            cls = idmap[ann["category_id"]]
            # clamp just in case
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            ww = max(0.0, min(1.0, ww))
            hh = max(0.0, min(1.0, hh))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

        out_path.write_text("\n".join(lines))
        wrote += 1
    return wrote

def write_dataset_yaml(out_dir: Path, names):
    # YOLO data YAML pointing to images dirs; labels are discovered automatically under labels/
    yml = [
        f'path: {out_dir}',
        'train: images/train',
        'val: images/val',
        'test: images/test',
        'names:'
    ]
    for i, n in enumerate(names):
        yml.append(f'  {i}: {n}')
    (out_dir / "dataset.yaml").write_text("\n".join(yml))

def main():
    parser = argparse.ArgumentParser(
        description="Split COCO JSON and build YOLO-format dataset (images/ + labels/ wrappers)."
    )
    parser.add_argument("json_path", type=str, help="Path to input COCO annotations JSON")
    parser.add_argument("images_root", type=str, help="Root directory that contains the images")
    parser.add_argument("out_dir", type=str, help="Output dataset root")
    parser.add_argument("--splits", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "symlink", "move"], default="symlink")
    parser.add_argument("--prefilter-missing", action="store_true")
    args = parser.parse_args()

    json_path, images_root, out_dir = Path(args.json_path), Path(args.images_root), Path(args.out_dir)
    if not json_path.exists():
        print(f"[ERR] JSON not found: {json_path}", file=sys.stderr); sys.exit(1)
    if not images_root.exists():
        print(f"[ERR] images_root not found: {images_root}", file=sys.stderr); sys.exit(1)

    coco = json.load(open(json_path))
    if args.prefilter_missing:
        coco, dropped = filter_out_missing_images(coco, images_root)
        if dropped:
            print(f"[INFO] Prefilter: dropped {dropped} images not found on disk.")

    train_ids, val_ids, test_ids = split_ids_uniform(coco["images"], args.splits, args.seed)
    idmap, names = make_cat_mapping(coco["categories"])

    # Make folders (YOLO standard)
    imgs_train = out_dir / "images" / "train"
    imgs_val   = out_dir / "images" / "val"
    imgs_test  = out_dir / "images" / "test"
    labels_root = out_dir / "labels"
    for d in (imgs_train, imgs_val, imgs_test, labels_root):
        d.mkdir(parents=True, exist_ok=True)

    # Place images per split
    ok_tr, miss_tr = materialize_split_files(coco, train_ids, images_root, imgs_train, args.mode)
    ok_va, miss_va = materialize_split_files(coco, val_ids,   images_root, imgs_val,   args.mode)
    ok_te, miss_te = materialize_split_files(coco, test_ids,  images_root, imgs_test,  args.mode)

    # Write YOLO labels per split
    wrote_tr = write_yolo_labels(coco, train_ids, labels_root, "train", idmap)
    wrote_va = write_yolo_labels(coco, val_ids,   labels_root, "val",   idmap)
    wrote_te = write_yolo_labels(coco, test_ids,  labels_root, "test",  idmap)

    # Write dataset.yaml (names included)
    write_dataset_yaml(out_dir, names)

    print("\n=== Summary ===")
    print(f"Train: images placed={ok_tr} (missing={miss_tr}), labels written for {wrote_tr} images")
    print(f"Val  : images placed={ok_va} (missing={miss_va}), labels written for {wrote_va} images")
    print(f"Test : images placed={ok_te} (missing={miss_te}), labels written for {wrote_te} images")
    print(f"\nDataset root: {out_dir}")
    print("Use this data YAML:")
    print(f"  data: {out_dir/'dataset.yaml'}")
    print("Then run YOLO with that data YAML.")

if __name__ == "__main__":
    main()
