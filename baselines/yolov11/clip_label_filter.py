#!/usr/bin/env python3

import argparse, os, shutil, glob, json, csv, sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
import clip
from PIL import Image
import yaml

# ------------------------ CLI ------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Filter YOLO train labels with CLIP (image-crop vs 'a photo of a {class}'). "
                    "Copies dataset to --dst and edits labels/train there."
    )
    p.add_argument("--src", required=True, help="Path to source dataset (has images/ and labels/)")
    p.add_argument("--dst", required=True, help="Destination dir (created; use --force to overwrite)")
    p.add_argument("--threshold", type=float, default=0.10, help="CLIP cosine threshold (default: 0.10)")
    p.add_argument("--pattern", default="*aug*.jpg", help="Which train images to check (glob, default: *aug*.jpg)")
    p.add_argument("--template", default="a photo of a {}", help="Text template for class names")
    p.add_argument("--names", default=None,
                   help="Path to class list (data.yaml with 'names', or JSON/lines). "
                        "If omitted, tries SRC/data.yaml.")
    p.add_argument("--model", default="ViT-B/32", help="CLIP model (e.g., 'ViT-B/32', 'ViT-L/14')")
    p.add_argument("--force", action="store_true", help="Overwrite destination if exists")
    p.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    p.add_argument("--report", default="clip_filter_report", help="Basename for report files")
    return p.parse_args()

# ------------------------ IO / Utils ------------------------

def load_names(src_dir: Path, names_arg: Optional[str]) -> List[str]:
    if names_arg:
        npath = Path(names_arg)
        if not npath.exists():
            sys.exit(f"--names not found: {npath}")
        if npath.suffix.lower() in [".yaml", ".yml"]:
            data = yaml.safe_load(npath.read_text())
            names = data.get("names", None)
            if names is None: sys.exit(f"'{npath}' has no 'names' field.")
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
            return list(names)
        else:
            try:
                data = json.loads(npath.read_text())
                if isinstance(data, dict):
                    data = [data[k] for k in sorted(data.keys(), key=lambda x: int(x))]
                return list(data)
            except json.JSONDecodeError:
                return [ln.strip() for ln in npath.read_text().splitlines() if ln.strip()]
    dpath = src_dir / "data.yaml"
    if dpath.exists():
        data = yaml.safe_load(dpath.read_text())
        names = data.get("names", None)
        if names is None: sys.exit("data.yaml found but no 'names' field.")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        return list(names)
    sys.exit("No class names found. Use --names or put data.yaml with 'names' in SRC.")

def ensure_dst(src: Path, dst: Path, force: bool, dry: bool):
    if dst.exists():
        if not force:
            sys.exit(f"Destination exists: {dst} (use --force to overwrite)")
        if not dry:
            shutil.rmtree(dst)
    if not dry:
        shutil.copytree(src, dst)

def yolo_to_xyxy(row, W, H) -> Tuple[int, int, int, int, int]:
    cls, cx, cy, w, h = row
    x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
    return cls, max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)

def read_yolo(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not txt_path.exists(): return []
    rows = []
    for line in txt_path.read_text().splitlines():
        p = line.strip().split()
        if len(p) != 5: continue
        cls, cx, cy, w, h = int(p[0]), *map(float, p[1:])
        rows.append((cls, cx, cy, w, h))
    return rows

def write_yolo(txt_path: Path, rows: List[Tuple[int, float, float, float, float]]):
    with open(txt_path, "w") as f:
        for cls, cx, cy, w, h in rows:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# ------------------------ CLIP ------------------------

class ClipScorer:
    def __init__(self, model_name: str, template: str, names: List[str]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        self.model.eval()
        prompts = [template.format(n) for n in names]
        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(self.device)
            tfeat = self.model.encode_text(tokens).float()
            self.text_feats = (tfeat / tfeat.norm(dim=-1, keepdim=True)).cpu()

    @torch.no_grad()
    def score_crop_vs_class(self, pil_img: Image.Image, class_id: int) -> float:
        # Guard against empty or tiny crops before calling torchvision transforms
        try:
            w, h = pil_img.size
        except Exception:
            return -1.0
        if w < 2 or h < 2:
            return -1.0

        # Preprocess can raise (e.g. resize division by zero) so catch exceptions
        try:
            img_t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        except Exception:
            return -1.0

        if img_t.numel() == 0:
            return -1.0
        if img_t.shape[-1] < 2 or img_t.shape[-2] < 2:
            return -1.0
        if img_t.std() == 0:
            return -1.0
        img_feat = self.model.encode_image(img_t).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ self.text_feats.to(self.device).T).squeeze(0)  # [num_classes]
        return float(sims[class_id].item())

# ------------------------ Core ------------------------

def process_train_split(dst: Path, pattern: str, threshold: float, scorer: ClipScorer,
                        dry: bool, names: List[str], report_base: str):
    img_dir = dst / "images" / "train"
    lab_dir = dst / "labels" / "train"
    if not img_dir.exists() or not lab_dir.exists():
        sys.exit("Expect images/train and labels/train in the dataset.")

    images = sorted(glob.glob(str(img_dir / pattern)))
    print(f"Train images matching pattern: {len(images)}")

    report_rows = []
    kept_total = removed_total = 0

    for ip in images:
        ipath = Path(ip)
        base = ipath.stem
        lpath = lab_dir / f"{base}.txt"
        if not lpath.exists():
            continue

        im = Image.open(ipath).convert("RGB")
        W, H = im.size
        rows = read_yolo(lpath)
        keep = []
        for (cls, cx, cy, w, h) in rows:
            cls_id, x1, y1, x2, y2 = yolo_to_xyxy((cls, cx, cy, w, h), W, H)
            crop = im.crop((x1, y1, x2, y2))
            s = scorer.score_crop_vs_class(crop, cls_id)
            drop = s < threshold
            report_rows.append({
                "image": ipath.name,
                "class_id": cls_id,
                "class_name": names[cls_id] if 0 <= cls_id < len(names) else str(cls_id),
                "cx": cx, "cy": cy, "w": w, "h": h,
                "score": round(s, 6),
                "dropped": int(drop),
            })
            if drop:
                removed_total += 1
            else:
                keep.append((cls, cx, cy, w, h)); kept_total += 1

        if not dry:
            write_yolo(lpath, keep)

    if not dry:
        json_path = dst / f"{report_base}.json"
        with open(json_path, "w") as f:
            json.dump(report_rows, f, indent=2)
        csv_path = dst / f"{report_base}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()) if report_rows else
                               ["image","class_id","class_name","cx","cy","w","h","score","dropped"])
            w.writeheader()
            for r in report_rows: w.writerow(r)

    print(f"Done. Kept labels: {kept_total}, removed: {removed_total}, images processed: {len(images)}")

# ------------------------ Main ------------------------

def main():
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not (src / "images").exists() or not (src / "labels").exists():
        sys.exit("SRC must have 'images' and 'labels' dirs (YOLO layout).")

    names = load_names(src, args.names)
    print(f"Classes: {len(names)} → {names[:5]}{' …' if len(names)>5 else ''}")

    ensure_dst(src, dst, args.force, args.dry_run)
    scorer = ClipScorer(args.model, args.template, names)

    print(f"Working only on TRAIN split in dst: {dst}")
    process_train_split(dst, args.pattern, args.threshold, scorer, args.dry_run, names, args.report)

if __name__ == "__main__":
    main()
