import argparse
import os
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Generate a photorealistic closeup image for a specified object class using SDXL."
    )

    parser.add_argument(
        "--object_class",
        type=str,
        required=True,
        help="The object class to render (e.g. 'spam can', 'coffee cup', 'banana peel').",
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="photoreal_closeup.png",
        help="Path to save the generated image. If generating multiple images, an index will be appended.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="How many variations to generate.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (must be multiple of 64).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (must be multiple of 64).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.5,
        help="Classifier-free guidance scale. 4–7 is a good range for SDXL.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Use -1 for random each time.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Build prompts from object class ---
    object_text = args.object_class.strip()

    base_prompt = f"a photo of a single {object_text}, entire object visible, centered"


    negative_prompt = "multiple objects, cropped, close-up, zoomed in, partial object, out of frame, cut off"




    print("Prompt:", base_prompt)
    print("Negative prompt:", negative_prompt)

    # --- Load SDXL txt2img pipeline ---
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    print(f"Loading SDXL model: {model_id}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
    )

    pipe = pipe.to(device)

    # Enable some lightweight memory optimizations (no xformers)
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing("auto")

    # --- Seed handling ---
    if args.seed is None or args.seed < 0:
        generator = None
        print("Using random seed for each image.")
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"Using fixed seed: {args.seed}")

    # --- Generate images ---
    os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)

    print("Generating images...")
    # You can generate num_images_per_prompt in a single call; here we loop for per-image seeds if needed
    for i in range(args.num_images):
        if generator is None:
            gen = torch.Generator(device=device).manual_seed(torch.seed())
        else:
            gen = generator

        out = pipe(
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            generator=gen,
        )

        image: Image.Image = out.images[0]

        if args.num_images == 1:
            out_path = args.output_image
        else:
            root, ext = os.path.splitext(args.output_image)
            out_path = f"{root}_{i:02d}{ext or '.png'}"

        image.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()