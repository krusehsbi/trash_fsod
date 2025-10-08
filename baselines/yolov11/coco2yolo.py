#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert COCO annotations to YOLO format.
"""

import json, os, shutil, argparse
from tqdm import tqdm

def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    return cx / img_w, cy / img_h, w / img_w, h / img_h

def convert_coco_to_yolo(coco_json, images_dir, out_dir, copy_images=True):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)

    with open(coco_json, "r") as f:
        coco = json.load(f)

    # map image_id → info
    img_info = {im["id"]: im for im in coco["images"]}

    # map category_id → contiguous YOLO id
    sorted_cats = sorted(coco["categories"], key=lambda x: x["id"])
    cat2yolo = {c["id"]: i for i, c in enumerate(sorted_cats)}
    # force all names to str to avoid TypeError
    names = [str(c.get("name", c["id"])) for c in sorted_cats]

    # group annotations
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_id, info in tqdm(img_info.items(), desc="Converting"):
        file_name = info["file_name"]
        img_w, img_h = info.get("width", 0), info.get("height", 0)
        anns = anns_by_img.get(img_id, [])

        # copy image
        src = os.path.join(images_dir, file_name)
        dst = os.path.join(out_dir, "images", os.path.basename(file_name))
        if copy_images:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        # label path
        label_path = os.path.join(
            out_dir, "labels", os.path.splitext(os.path.basename(file_name))[0] + ".txt"
        )

        with open(label_path, "w") as f:
            for a in anns:
                cid = cat2yolo[a["category_id"]]
                x, y, w, h = coco_to_yolo_bbox(a["bbox"], img_w, img_h)
                f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # names.txt for reference (always stringified)
    with open(os.path.join(out_dir, "names.txt"), "w") as f:
        f.write("\n".join(names))

    print(f"\n✅ Converted {len(img_info)} images → YOLO format.")
    print(f"Saved to: {out_dir}")
    print(f"Classes ({len(names)}): {names}")

def main():
    ap = argparse.ArgumentParser("Convert COCO dataset to YOLO format.")
    ap.add_argument("--coco-json", required=True, help="Path to COCO JSON")
    ap.add_argument("--images-dir", required=True, help="Path to image directory")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--no-copy", action="store_true", help="Do not copy images (just write labels)")
    args = ap.parse_args()

    convert_coco_to_yolo(args.coco_json, args.images_dir, args.out_dir, copy_images=not args.no_copy)

if __name__ == "__main__":
    main()
