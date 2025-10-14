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
      --model runwayml/stable-diffusion-v1-5 \
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

import torch
from diffusers import StableDiffusionImg2ImgPipeline

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, type=str,
                        help="Original YOLO dataset root with images/ and labels/ subfolders.")
    parser.add_argument("--out_root", required=True, type=str,
                        help="Completely new output root directory for augmented dataset.")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    args = parser.parse_args()

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

    print(f"Loading diffusion pipeline {args.model} ...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        safety_checker=None,         # skip NSFW head entirely
        feature_extractor=None,      # skip unused extractor
        low_cpu_mem_usage=False,     # <- important: don't trigger offload path
        device_map=None              # <- make sure transformers doesn't try to shard/offload
    ).to("cuda")

    image_paths = sorted(list(img_in.glob("**/*")))
    image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    print(f"Found {len(image_paths)} images to process.")

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

        # always copy base image + label
        shutil.copy2(img_path, out_img_path)
        if label_path.exists():
            shutil.copy2(label_path, out_lbl_path)

        if not boxes:
            continue

        for bidx, (cls_id, x1, y1, x2, y2) in enumerate(boxes):
            crop = img.crop((x1, y1, x2, y2))
            bbox_w, bbox_h = crop.size
            square_init, off_info = pil_to_square(crop, 512)

            for k in range(args.instances_per_box):
                gen_seed = args.seed + img_idx * 10000 + bidx * 100 + k
                generator = torch.Generator(device=args.device).manual_seed(gen_seed)

                prompt = args.default_prompt
                neg_prompt = args.negative_prompt

                result = pipe(
                    prompt=prompt,
                    image=square_init,
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=neg_prompt,
                    num_inference_steps=30,
                    generator=generator,
                )
                gen_sq = result.images[0]
                gen_region = square_to_bbox_region(gen_sq, off_info, bbox_w, bbox_h)

                # Paste
                pasted = img.copy()
                if args.use_rembg and HAS_REMBG:
                    rgba = remove_bg_rgba(gen_region)
                    pasted = paste_with_alpha(pasted, rgba, x1, y1)
                else:
                    pasted.paste(gen_region, (x1, y1))

                aug_name = f"{img_path.stem}_aug_b{bidx:02d}_k{k:02d}{img_path.suffix}"
                aug_out_path = img_out / rel_path.parent / aug_name
                ensure_dir(aug_out_path.parent)
                pasted.save(aug_out_path, quality=95)

                # duplicate original labels for augmented image
                if label_path.exists():
                    with open(label_path, "r") as src, open(lbl_out / (aug_out_path.stem + ".txt"), "w") as dst:
                        dst.write(src.read())

    print(f"\n Augmentation complete!")
    print(f"New dataset written to: {out_root}")


if __name__ == "__main__":
    
    print("PY:", sys.executable)
    print("diffusers:", diffusers.__version__)
    print("transformers:", transformers.__version__)
    print("huggingface_hub:", huggingface_hub.__version__)
    print("accelerate:", accelerate.__version__)
    print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
    import transformers, inspect
    print("transformers file:", transformers.__file__)
    from transformers.models.clip.modeling_clip import CLIPTextModel
    print("CLIPTextModel from:", CLIPTextModel.__module__)
    print("CLIPTextModel.__init__ args:", CLIPTextModel.__init__.__code__.co_varnames)
    main()