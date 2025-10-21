#!/usr/bin/env python3
"""
Crop objects from YOLO-format labels.
Usage:
  python crop_yolo.py --img_dir images --lbl_dir labels --out_dir crops
"""

import os
import argparse
from pathlib import Path
from PIL import Image

def crop_yolo(img_path, lbl_path, out_dir):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # read label lines
    if not os.path.exists(lbl_path):
        return
    with open(lbl_path) as f:
        lines = [l.strip().split() for l in f if l.strip()]

    for i, parts in enumerate(lines):
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = map(float, parts)
        x1 = int((cx - w/2) * W)
        y1 = int((cy - h/2) * H)
        x2 = int((cx + w/2) * W)
        y2 = int((cy + h/2) * H)
        crop = img.crop((x1, y1, x2, y2))

        # save crop
        out_path = Path(out_dir) / f"{img_path.stem}_cls{int(cls)}_{i:02d}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, quality=95)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--lbl_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    img_paths = list(Path(args.img_dir).rglob("*.jpg")) + list(Path(args.img_dir).rglob("*.JPG"))
    for img_path in img_paths:
        lbl_path = Path(args.lbl_dir) / img_path.with_suffix(".txt").name
        crop_yolo(img_path, lbl_path, args.out_dir)

if __name__ == "__main__":
    main()