#!/usr/bin/env python3
import argparse, os, random, math, hashlib
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

def seed_all(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_pipe(model_id, device, fp16=True):
    dtype = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype==torch.float16 else None
    )
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()  # good default; uses GPU when needed
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    return pipe.to(device)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def center_pad_to_multiple(img: Image.Image, multiple=8, fill=(0,0,0,0)):
    # SDXL works best with dims multiple of 8
    w, h = img.size
    nw = math.ceil(w/multiple)*multiple
    nh = math.ceil(h/multiple)*multiple
    if (nw, nh) == (w, h):
        return img
    canvas = Image.new(img.mode, (nw, nh), fill)
    canvas.paste(img, ((nw - w)//2, (nh - h)//2))
    return canvas

def resize_bg(bg: Image.Image, target_size):
    # Letterbox-style fit then center-crop to target
    tw, th = target_size
    bw, bh = bg.size
    scale = max(tw/bw, th/bh)
    new_size = (max(1, int(bw*scale)), max(1, int(bh*scale)))
    bg = bg.resize(new_size, Image.BICUBIC)
    left = (bg.width - tw)//2
    top  = (bg.height - th)//2
    return bg.crop((left, top, left+tw, top+th))

def list_image_files(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def main():
    parser = argparse.ArgumentParser(description="Generate SDXL variations for object renders (recursive).")
    parser.add_argument("--input_dir", required=True, type=Path, help="Root folder with PNG renders (walks subfolders).")
    parser.add_argument("--output_dir", required=True, type=Path, help="Where to save generated images.")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--variants", type=int, default=5, help="Number of variants per input image.")
    parser.add_argument("--prompt", type=str, default="highly detailed product photo, studio lighting, soft shadows, sharp focus")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, deformed, extra parts, watermark, text, logo")
    parser.add_argument("--strength", type=float, default=0.35, help="Denoising strength (0.2–0.5 keeps identity).")
    parser.add_argument("--guidance", type=float, default=5.5, help="CFG scale.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234, help="Base seed (per-variant offset).")
    parser.add_argument("--keep_alpha", action="store_true", help="Reapply original alpha to keep transparent background.")
    parser.add_argument("--backgrounds_dir", type=Path, default=None, help="If set, composite variants over random backgrounds from this folder.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_size", type=int, default=1024, help="Max side length fed to SDXL (downscales proportionally).")
    parser.add_argument("--save_jpeg", action="store_true", help="Also save an 8-bit JPEG alongside PNG (useful for some pipelines).")
    args = parser.parse_args()

    seed_all(args.seed)
    ensure_dir(args.output_dir)
    pipe = load_pipe(args.model_id, args.device, fp16=True)

    bg_files = []
    if args.backgrounds_dir:
        bg_files = [p for p in list_image_files(args.backgrounds_dir)]
        if not bg_files:
            print(f"[WARN] backgrounds_dir provided but no images found at {args.backgrounds_dir}")

    input_files = list(list_image_files(args.input_dir))
    if not input_files:
        print("[ERROR] No input images found.")
        return

    for src in tqdm(input_files, desc="Processing"):
        rel = src.relative_to(args.input_dir)
        out_dir = args.output_dir / rel.parent
        ensure_dir(out_dir)

        # Load input (expecting PNG with alpha, but handle others)
        img = Image.open(src).convert("RGBA")
        # Keep a copy of alpha for later compositing
        alpha = img.split()[-1]

        # Downscale to max_size for SDXL if needed (maintain aspect ratio)
        w, h = img.size
        scale = min(1.0, args.max_size / max(w, h))
        work = img if scale == 1.0 else img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        # SDXL expects RGB; give it a neutral canvas behind the object for better results
        work_rgb = Image.new("RGB", work.size, (245, 245, 245))
        work_rgb.paste(work.convert("RGB"), mask=work.split()[-1])
        work_rgb = center_pad_to_multiple(work_rgb, multiple=8, fill=(245,245,245))
        padded_alpha = center_pad_to_multiple(work.split()[-1], multiple=8, fill=0)

        # Create per-file id for reproducibility across runs
        file_hash = int(hashlib.sha1(str(rel).encode()).hexdigest(), 16) % (2**31)

        for i in range(args.variants):
            # Variant seed: base + file_hash + index
            variant_seed = (args.seed or 0) + file_hash + i
            generator = torch.Generator(device=args.device).manual_seed(variant_seed)

            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=work_rgb,
                strength=args.strength,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=generator,
            ).images[0]

            # Remove any background and restore transparency if requested
            if args.keep_alpha or args.backgrounds_dir:
                # Unpad to original working size
                if result.size != work_rgb.size:
                    result = result.resize(work_rgb.size, Image.LANCZOS)

                # Reapply alpha from input (aligned to padded canvas)
                result_rgba = result.convert("RGBA")
                result_rgba.putalpha(padded_alpha)

                final_rgba = result_rgba

                # Optional: composite onto a random background
                if bg_files:
                    bg = Image.open(random.choice(bg_files)).convert("RGB")
                    bg = resize_bg(bg, final_rgba.size)
                    bg = bg.convert("RGBA")
                    composed = bg.copy()
                    composed.paste(final_rgba, mask=final_rgba.split()[-1])
                    save_img = composed.convert("RGB")  # dataset-friendly JPG too
                else:
                    # Keep transparent
                    save_img = final_rgba

            else:
                # No alpha restoration; just return SDXL output (RGB) with a neutral bg
                save_img = result

            # Save at source-mirrored path
            name = src.stem + f"_var{i+1}_seed{variant_seed}"
            # If we downscaled for the model, upscale result back to original size to match masks/datasets
            if save_img.size != img.size:
                save_img = save_img.resize(img.size, Image.LANCZOS)

            out_png = out_dir / f"{name}.png"
            save_img.save(out_png)

            if args.save_jpeg:
                out_jpg = out_dir / f"{name}.jpg"
                save_img.convert("RGB").save(out_jpg, quality=95)

    print("Done.")

if __name__ == "__main__":
    main()
