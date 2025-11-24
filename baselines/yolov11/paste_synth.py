import argparse
import random
import shutil
from pathlib import Path

from rembg import remove
from PIL import Image


def copy_dataset(src_root: Path, dst_root: Path):
    """
    Copy entire dataset tree from src_root to dst_root.
    Keeps structure (images/, labels/, yaml, etc.).
    """
    print(f"Copying dataset from {src_root} to {dst_root}...")
    shutil.copytree(src_root, dst_root, dirs_exist_ok=True)
    print("Copy finished.")


def load_yolo_labels(label_file: Path):
    if not label_file or not label_file.exists():
        return []
    with open(label_file, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def save_yolo_labels(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def find_image_files(root: Path):
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(root.rglob(ext))
    return images


def build_label_lookup(labels_root: Path):
    """
    Build dict: {"image_stem": label_path} for all .txt files.
    """
    lookup = {}
    for txt in labels_root.rglob("*.txt"):
        lookup[txt.stem] = txt
    return lookup


def get_synthetic_cutouts_by_class(synth_root: Path):
    """
    Expected structure:
        synthetic/
            0/
            1/
            2/
    Returns: {class_id: [cutout_images]}
    """
    class_cutouts = {}

    for class_folder in sorted(synth_root.iterdir()):
        if not class_folder.is_dir():
            continue

        try:
            class_id = int(class_folder.name)
        except ValueError:
            # ignore non-numeric folders
            continue

        cutouts = []
        images = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            images.extend(class_folder.glob(ext))

        prog=0
        for p in images:
            prog = prog+1
            print(prog)
            img = Image.open(p).convert("RGBA")
            no_bg = remove(img)

            alpha = no_bg.split()[-1]
            bbox = alpha.getbbox()
            if bbox is None:
                continue

            cutouts.append(no_bg.crop(bbox))

        if cutouts:
            class_cutouts[class_id] = cutouts

    return class_cutouts


def random_scale(cutout: Image.Image, bg_w, bg_h, min_scale, max_scale):
    scale = random.uniform(min_scale, max_scale)
    ref = min(bg_w, bg_h)
    target_size = int(ref * scale)

    w, h = cutout.size
    if w >= h:
        new_w = target_size
        new_h = int(target_size * (h / w))
    else:
        new_h = target_size
        new_w = int(target_size * (w / h))

    if new_w < 1 or new_h < 1:
        return None

    return cutout.resize((new_w, new_h), Image.LANCZOS)


def random_position(cut: Image.Image, bg_w, bg_h):
    w, h = cut.size
    if w > bg_w or h > bg_h:
        return None
    x = random.randint(0, bg_w - w)
    y = random.randint(0, bg_h - h)
    return x, y


def abs_to_yolo(x_min, y_min, x_max, y_max, W, H):
    w = x_max - x_min
    h = y_max - y_min
    xc = x_min + w / 2
    yc = y_min + h / 2
    return xc / W, yc / H, w / W, h / H


def main():
    parser = argparse.ArgumentParser(
        description="Copy full YOLO dataset and augment the train set with synthetic objects."
    )

    parser.add_argument(
        "--src-root",
        required=True,
        help="Source dataset root (contains images/, labels/, yaml, etc.)",
    )
    parser.add_argument(
        "--dst-root",
        required=True,
        help="Destination dataset root (will be created/overwritten).",
    )
    parser.add_argument(
        "--synthetic-dir",
        required=True,
        help="Folder with synthetic images in class folders 0,1,2,...",
    )
    parser.add_argument(
        "--subset",
        default="train",
        help="Subset to augment inside dst-root/images and dst-root/labels (default: train)",
    )
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--min-scale", type=float, default=0.15)
    parser.add_argument("--max-scale", type=float, default=0.40)

    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    synth_root = Path(args.synthetic_dir)

    # 1) Copy full dataset tree
    copy_dataset(src_root, dst_root)

    # 2) Paths inside the copied dataset
    images_root = dst_root / "images"
    labels_root = dst_root / "labels"

    img_subset_root = images_root / args.subset
    lbl_subset_root = labels_root / args.subset

    if not img_subset_root.exists():
        print(f"Subset images folder does not exist: {img_subset_root}")
        return
    if not lbl_subset_root.exists():
        print(f"Subset labels folder does not exist: {lbl_subset_root}")
        return

    # Background images from the copied train set
    images = find_image_files(img_subset_root)
    if not images:
        print(f"No images found under {img_subset_root}")
        return

    # Label lookup inside copied train labels
    label_lookup = build_label_lookup(lbl_subset_root)

    # Synthetic cutouts by class
    class_cutouts = get_synthetic_cutouts_by_class(synth_root)
    if not class_cutouts:
        print("No valid synthetic class folders (0,1,2,...) with usable images.")
        return

    print(f"Found synthetic classes: {list(class_cutouts.keys())}")
    print(f"Augmenting subset '{args.subset}' in copied dataset ({len(images)} base images)...")

    for img_path in images:
        bg = Image.open(img_path).convert("RGB")
        bg_w, bg_h = bg.size
        bg_rgba = bg.convert("RGBA")

        label_file = label_lookup.get(img_path.stem)
        labels = load_yolo_labels(label_file)

        n_objects = random.randint(args.min_objects, args.max_objects)

        for _ in range(n_objects):
            class_id = random.choice(list(class_cutouts.keys()))
            cut = random.choice(class_cutouts[class_id])

            cut_resized = random_scale(cut, bg_w, bg_h, args.min_scale, args.max_scale)
            if cut_resized is None:
                continue

            pos = random_position(cut_resized, bg_w, bg_h)
            if pos is None:
                continue

            x, y = pos
            w_obj, h_obj = cut_resized.size

            bg_rgba.paste(cut_resized, (x, y), cut_resized)

            x_c, y_c, bw, bh = abs_to_yolo(
                x, y, x + w_obj, y + h_obj, bg_w, bg_h
            )
            labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        # augmented file name: original_stem_aug
        rel = img_path.relative_to(img_subset_root)
        rel_parent = rel.parent
        stem = rel.stem
        suffix = rel.suffix

        aug_img_name = f"{stem}_aug{suffix}"
        aug_lbl_name = f"{stem}_aug.txt"

        out_img_path = img_subset_root / rel_parent / aug_img_name
        out_lbl_path = lbl_subset_root / rel_parent / aug_lbl_name

        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

        bg_rgba.convert("RGB").save(out_img_path)
        save_yolo_labels(out_lbl_path, labels)

        print(f"[{args.subset}] {rel} -> {rel_parent / aug_img_name}")


if __name__ == "__main__":
    main()
