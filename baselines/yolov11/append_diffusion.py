#!/usr/bin/env python3
"""
YOLO few-shot augmentation with diffusion (img2img) per instance.

For each YOLO bbox:
  1) crop the object
  2) generate N variants via Stable Diffusion img2img (guided by the crop)
  3) paste each variant into a COPY of the original image at the same bbox
  4) save to a *new dataset root* (so the original stays intact)
  5) replicate YOLO label files alongside generated images

Usage:
  python yolo_diffusion_augment.py \
      --dataset_root /path/to/original_dataset \
      --out_root /path/to/new_augmented_dataset \
      --instances_per_box 3 \
      --use_rembg

Requirements:
  pip install diffusers transformers accelerate torch torchvision pillow numpy rembg
"""

import argparse
import os
import shutil
from pathlib import Path
import random
import numpy as np
import sys, diffusers, transformers, huggingface_hub, accelerate, torch
from PIL import Image
from PIL import ImageFilter
import json

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except Exception:
    HAS_REMBG = False


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            cls = int(cls)
            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h
            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = min(img_w, int(cx + w / 2))
            y2 = min(img_h, int(cy + h / 2))
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def pil_to_square(pil_img, target=512, pad_color=(255, 255, 255)):
    w, h = pil_img.size
    scale = min(target / w, target / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target, target), pad_color)
    off_x = (target - new_w) // 2
    off_y = (target - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas, (off_x, off_y, new_w, new_h)


def square_to_bbox_region(square_img, off_info, bbox_w, bbox_h):
    off_x, off_y, new_w, new_h = off_info
    region = square_img.crop((off_x, off_y, off_x + new_w, off_y + new_h))
    region = region.resize((bbox_w, bbox_h), Image.LANCZOS)
    return region


def remove_bg_rgba(pil_img):
    if not HAS_REMBG:
        return pil_img.convert("RGBA")
    arr = np.array(pil_img.convert("RGBA"))
    cut = rembg_remove(arr)
    return Image.fromarray(cut).convert("RGBA")


def paste_with_alpha(base_img, overlay_rgba, x1, y1):
    base = base_img.convert("RGBA")
    W, H = base.size
    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    tmp.paste(overlay_rgba, (x1, y1), overlay_rgba)
    out = Image.alpha_composite(base, tmp)
    return out.convert("RGB")

def _expand_with_context(img, x1, y1, x2, y2, factor=1.75):
    W, H = img.size
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nw, nh = int(w * factor), int(h * factor)
    nx1, ny1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
    nx2, ny2 = min(W, int(cx + nw / 2)), min(H, int(cy + nh / 2))
    return nx1, ny1, nx2, ny2

def prep_highres_init(img, x1, y1, x2, y2, small_thr=64, gen_target=1024, context_factor=1.75):
    """
    If bbox short edge < small_thr:
      - take a context crop around bbox
      - upscale to gen_target on the long side
      - pad to square
    Returns:
      square_init (PIL), off_info, is_small(bool), ctx_box or None
    """
    bw, bh = x2 - x1, y2 - y1
    short_edge = min(bw, bh)
    if short_edge >= small_thr:
        crop = img.crop((x1, y1, x2, y2))
        square_init, off_info = pil_to_square(crop, target=1024)  # SDXL native
        return square_init, off_info, False, None

    # small object path
    ex1, ey1, ex2, ey2 = _expand_with_context(img, x1, y1, x2, y2, factor=context_factor)
    region = img.crop((ex1, ey1, ex2, ey2))
    rw, rh = region.size
    scale = gen_target / max(rw, rh)
    up_w, up_h = max(64, int(rw * scale)), max(64, int(rh * scale))
    region_up = region.resize((up_w, up_h), Image.LANCZOS)
    side = max(up_w, up_h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    off_x, off_y = (side - up_w) // 2, (side - up_h) // 2
    canvas.paste(region_up, (off_x, off_y))
    off_info = (off_x, off_y, up_w, up_h)
    return canvas, off_info, True, (ex1, ey1, ex2, ey2)

def map_back_highres(gen_sq, off_info, ctx_box, x1, y1, x2, y2):
    """
    Map the generated high-res+context result back to the original bbox size.
    Returns: gen_region (PIL) with size (bbox_w, bbox_h)
    """
    bw, bh = x2 - x1, y2 - y1
    off_x, off_y, up_w, up_h = off_info
    ex1, ey1, ex2, ey2 = ctx_box

    # remove square padding
    region_no_pad = gen_sq.crop((off_x, off_y, off_x + up_w, off_y + up_h))

    # map bbox coords from original context crop -> upscaled region
    sx = up_w / float(ex2 - ex1 + 1e-6)
    sy = up_h / float(ey2 - ey1 + 1e-6)
    rel_x1 = int((x1 - ex1) * sx); rel_y1 = int((y1 - ey1) * sy)
    rel_x2 = int((x2 - ex1) * sx); rel_y2 = int((y2 - ey1) * sy)
    rel_x1 = max(0, min(up_w, rel_x1)); rel_y1 = max(0, min(up_h, rel_y1))
    rel_x2 = max(0, min(up_w, rel_x2)); rel_y2 = max(0, min(up_h, rel_y2))

    gen_region_big = region_no_pad.crop((rel_x1, rel_y1, rel_x2, rel_y2))
    return gen_region_big.resize((bw, bh), Image.LANCZOS)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, type=str,
                        help="Original YOLO dataset root with images/ and labels/ subfolders.")
    parser.add_argument("--out_root", required=True, type=str,
                        help="Completely new output root directory for augmented dataset.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to JSON file containing class-specific prompts.")
    parser.add_argument("--instances_per_box", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_rembg", action="store_true",
                        help="Use background removal before pasting.")
    parser.add_argument("--default_prompt", type=str,
                        default="a high quality, photorealistic object")
    parser.add_argument("--negative_prompt", type=str,
                        default="low quality, blurry, text, watermark")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--small_thr", type=int, default=256,
                    help="If min(bbox_w,bbox_h) < small_thr, use high-res generation path.")
    parser.add_argument("--gen_target_small", type=int, default=1024,
                        help="Generation target side for small-object path (use 768 for SD1.5, 1024 for SDXL).")
    parser.add_argument("--context_factor", type=float, default=1.75,
                        help="Context expansion factor around small bboxes before upscaling.")

    args = parser.parse_args()

    if args.prompt_file is not None and Path(args.prompt_file).exists():
        with open(args.prompt_file, "r") as f:
            PROMPTS = json.load(f)
        NEG_PROMPT = PROMPTS.get("negative", args.negative_prompt)
        print(f"Loaded {len(PROMPTS)} prompts from {args.prompt_file}")
    else:
        PROMPTS = {}
        NEG_PROMPT = args.negative_prompt

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)

    img_in = dataset_root / "images"
    lbl_in = dataset_root / "labels"
    img_out = out_root / "images"
    lbl_out = out_root / "labels"

    if out_root.exists() and not args.overwrite:
        raise SystemExit(f"Output root '{out_root}' exists. Use --overwrite to replace it.")
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)

    ensure_dir(img_out)
    ensure_dir(lbl_out)

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        safety_checker=None,
        add_watermarker=False,
        low_cpu_mem_usage=False,
        device_map=None
    ).to(args.device)

    # ----- split-aware IO setup -----
    img_in_root = dataset_root / "images"
    lbl_in_root = dataset_root / "labels"
    img_out_root = out_root / "images"
    lbl_out_root = out_root / "labels"

    # detect split layout
    has_splits = any((img_in_root / s).exists() for s in ["train", "val", "test"])

    def copy_split(src_img_dir: Path, src_lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path):
        if not src_img_dir.exists():
            return
        ensure_dir(dst_img_dir)
        ensure_dir(dst_lbl_dir)
        # copy images
        for p in src_img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rel = p.relative_to(src_img_dir)
                ensure_dir((dst_img_dir / rel).parent)
                shutil.copy2(p, dst_img_dir / rel)
                # copy matching label if exists
                lp = src_lbl_dir / rel.with_suffix(".txt")
                if lp.exists():
                    ensure_dir((dst_lbl_dir / rel.with_suffix(".txt")).parent)
                    shutil.copy2(lp, dst_lbl_dir / rel.with_suffix(".txt"))

    # ------------------------------------------------------------
    # AUGMENT train; COPY val/test (if present). If no splits, augment all.
    # ------------------------------------------------------------
    if has_splits:
        splits = ["train", "val", "test"]
    else:
        splits = ["train"]  # treat entire dataset as train (no val/test present)

    for split in splits:
        img_in = img_in_root / split if has_splits else img_in_root
        lbl_in = lbl_in_root / split if has_splits else lbl_in_root
        img_out = img_out_root / split if has_splits else img_out_root
        lbl_out = lbl_out_root / split if has_splits else lbl_out_root

        ensure_dir(img_out)
        ensure_dir(lbl_out)

        if split in ["val", "test"] and has_splits:
            print(f"[COPY] {split}: copying without augmentation …")
            copy_split(img_in, lbl_in, img_out, lbl_out)
            continue

        # ---------- TRAIN AUGMENTATION ----------
        image_paths = sorted(p for p in img_in.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"])
        print(f"[AUGMENT] {split}: found {len(image_paths)} images to process.")

        for img_idx, img_path in enumerate(image_paths):
            rel_path = img_path.relative_to(img_in)
            out_img_path = img_out / rel_path
            out_lbl_path = lbl_out / rel_path.with_suffix(".txt")
            ensure_dir(out_img_path.parent)
            ensure_dir(out_lbl_path.parent)

            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            label_path = lbl_in / rel_path.with_suffix(".txt")
            boxes = load_yolo_boxes(label_path, W, H)

            # always copy base image + label (unaltered)
            shutil.copy2(img_path, out_img_path)
            if label_path.exists():
                ensure_dir(out_lbl_path.parent)
                shutil.copy2(label_path, out_lbl_path)

            if not boxes:
                continue

            # Create exactly K augmented versions of THIS image.
            for k in range(args.instances_per_box):

                print(f"[{img_idx+1}/{len(image_paths)}] {rel_path} -> aug k={k}")

                gen_seed_base = args.seed + img_idx * 10000 + k * 1000
                # start from the original each time; we will modify ALL objects into this copy
                pasted_k = img.copy()

                for bidx, (cls_id, x1, y1, x2, y2) in enumerate(boxes):
                    # --- prepare init (small-object aware; fall back to your 1024-square if you don't use these helpers) ---
                    crop = img.crop((x1, y1, x2, y2))
                    bbox_w, bbox_h = crop.size

                    square_init, off_info, is_small, ctx_box = prep_highres_init(
                        img, x1, y1, x2, y2,
                        small_thr=args.small_thr,              # e.g., 64
                        gen_target=args.gen_target_small,      # e.g., 1024 for SDXL
                        context_factor=args.context_factor     # e.g., 1.75
                    )

                    # prompts
                    opts = PROMPTS.get(str(cls_id), args.default_prompt)
                    prompt = random.choice(opts) if isinstance(opts, list) else opts
                    neg_prompt = NEG_PROMPT

                    # per-object deterministic seed for this k
                    generator = torch.Generator(device=args.device).manual_seed(gen_seed_base + bidx)

                    # use lower strength for tiny objects to keep identity
                    run_strength = min(args.strength, 0.35) if is_small else args.strength

                    # --- generate ---
                    result = pipe(
                        prompt=prompt,
                        image=square_init,
                        strength=run_strength,
                        guidance_scale=6.0,          # SDXL sweet spot ~5–7
                        negative_prompt=neg_prompt,
                        num_inference_steps=35,
                        generator=generator,
                    )
                    gen_sq = result.images[0]

                    # map back to bbox size
                    if is_small:
                        gen_region = map_back_highres(gen_sq, off_info, ctx_box, x1, y1, x2, y2)
                    else:
                        gen_region = square_to_bbox_region(gen_sq, off_info, bbox_w, bbox_h)

                    # paste into THIS image's Kth variant
                    if args.use_rembg and HAS_REMBG:
                        rgba = remove_bg_rgba(gen_region)
                        pasted_k = paste_with_alpha(pasted_k, rgba, x1, y1)
                    else:
                        pasted_k.paste(gen_region, (x1, y1))

                # after all objects are updated, save ONE augmented image for this k
                aug_name = f"{img_path.stem}_aug_k{k:02d}{img_path.suffix}"
                aug_out_path = img_out / rel_path.parent / aug_name
                ensure_dir(aug_out_path.parent)
                pasted_k.save(aug_out_path, quality=95)

                # labels unchanged geometrically → copy original with new name
                if label_path.exists():
                    with open(label_path, "r") as src, open(lbl_out / (aug_out_path.stem + ".txt"), "w") as dst:
                        dst.write(src.read())

    print(f"\n Augmentation complete!")
    print(f"New dataset written to: {out_root}")


if __name__ == "__main__":
    main()