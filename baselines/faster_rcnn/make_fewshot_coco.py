#!/usr/bin/env python3
import argparse, json, os, random, re, shutil, sys
from pathlib import Path
from collections import defaultdict
from difflib import get_close_matches
from typing import Dict, List, Set, Tuple, Optional

# ----------------------------- utils -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def place_image(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if mode == "none":
        return
    if mode == "symlink":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            rel = Path(os.path.relpath(src, start=dst.parent))
            dst.symlink_to(rel)
        except Exception:
            shutil.copy2(src, dst)
    elif mode == "copy":
        if not dst.exists():
            shutil.copy2(src, dst)
    elif mode == "move":
        if not dst.exists():
            shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"unknown mode: {mode}")

def rel_from_images_root(file_name: str, images_root: Path) -> Path:
    fn = file_name.lstrip("/")
    cand = images_root / fn
    if cand.exists():
        try:
            return cand.relative_to(images_root)
        except Exception:
            pass
    base = Path(fn).name
    for p in images_root.rglob(base):
        if p.is_file():
            try:
                return p.relative_to(images_root)
            except Exception:
                return Path(base)
    return Path(fn)

def norm(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"\s+", " ", s)
    return s

# ----------------------------- coco I/O -----------------------------
def load_coco(p: Path) -> Dict:
    data = json.loads(p.read_text(encoding="utf-8"))
    for im in data["images"]:
        im["id"] = int(im["id"])
    for ann in data["annotations"]:
        ann["id"] = int(ann.get("id", 0))
        ann["image_id"] = int(ann["image_id"])
        ann["category_id"] = int(ann["category_id"])
    for c in data["categories"]:
        c["id"] = int(c["id"])
    return data

def build_super_indexes(categories: List[Dict]):
    super2catids: Dict[str, Set[int]] = defaultdict(set)
    for c in categories:
        super2catids[c.get("supercategory", "")].add(int(c["id"]))
    return super2catids

# ------------------------ selection / splitting ------------------------
def pick_k_per_super(
    coco: Dict, ordered_supers: List[str], K: int, seed: int
) -> Tuple[Set[int], Dict[int, List[int]], List[int], Dict[str, int]]:
    random.seed(seed)
    catid2super = {int(c["id"]): c.get("supercategory", "") for c in coco["categories"]}
    groups = defaultdict(list)  # super -> list of anns
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) == 1 or "bbox" not in a:  # skip crowds
            continue
        sname = catid2super.get(int(a["category_id"]), "")
        groups[sname].append(a)

    selected_ann_ids: Set[int] = set()
    per_super_available: Dict[str, int] = {}
    for s in ordered_supers:
        lst = groups.get(s, [])
        per_super_available[s] = len(lst)
        random.shuffle(lst)
        take = lst[:K] if len(lst) >= K else lst
        selected_ann_ids.update(int(a["id"]) for a in take)

    train_img_ids: Set[int] = set()
    train_ann_ids_per_img: Dict[int, List[int]] = defaultdict(list)
    for a in coco["annotations"]:
        if int(a["id"]) in selected_ann_ids:
            iid = int(a["image_id"])
            train_img_ids.add(iid)
            train_ann_ids_per_img[iid].append(int(a["id"]))

    all_img_ids = {int(im["id"]) for im in coco["images"]}
    remaining_img_ids = sorted(list(all_img_ids - train_img_ids))
    return train_img_ids, train_ann_ids_per_img, remaining_img_ids, per_super_available

def split_remaining(remaining_img_ids: List[int], val_ratio: float, seed: int) -> Tuple[Set[int], Set[int]]:
    random.seed(seed)
    ids = remaining_img_ids[:]
    random.shuffle(ids)
    n_val = int(val_ratio * len(ids))
    return set(ids[:n_val]), set(ids[n_val:])

# -------------------------- subset/remap --------------------------
def subset_coco(
    coco: Dict,
    keep_img_ids: Set[int],
    allowed_cat_ids: Set[int],
    train_ann_filter: Optional[Dict[int, List[int]]] = None,
) -> Dict:
    images = [im for im in coco["images"] if int(im["id"]) in keep_img_ids]
    keep_set = {int(im["id"]) for im in images}
    anns = []
    for a in coco["annotations"]:
        if int(a["image_id"]) not in keep_set:
            continue
        if train_ann_filter is not None:
            chosen = train_ann_filter.get(int(a["image_id"]), [])
            if int(a["id"]) not in chosen:
                continue
        if int(a["category_id"]) not in allowed_cat_ids:
            continue
        anns.append(a)
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": coco["categories"],  # replaced later
    }

def remap_to_supercats(ds: Dict, coco: Dict, ordered_supers: List[str]) -> Dict:
    super2newid = {name: i + 1 for i, name in enumerate(ordered_supers)}
    catid2super = {int(c["id"]): c.get("supercategory", "") for c in coco["categories"]}
    new_anns = []
    for a in ds["annotations"]:
        sname = catid2super.get(int(a["category_id"]), "")
        if sname not in super2newid:
            continue
        b = a.copy()
        b["category_id"] = super2newid[sname]
        new_anns.append(b)
    ds["annotations"] = new_anns
    ds["categories"] = [{"id": i + 1, "name": name, "supercategory": name} for i, name in enumerate(ordered_supers)]
    return ds

def mirror_images(split_name: str, images: List[Dict], images_root: Path, out_root: Path, mode: str):
    if mode == "none":
        return
    out_dir = out_root / "images" / split_name
    for im in images:
        rel = rel_from_images_root(im["file_name"], images_root)
        src = images_root / rel
        dst = out_dir / rel
        place_image(src, dst, mode)

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Few-shot COCO builder using SUPER-CATEGORIES only.")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", required=True, help="Comma-separated SUPER-CATEGORY names in desired order")
    ap.add_argument("--k", type=int, required=True, help="K annotations per supercategory for TRAIN")
    ap.add_argument("--val-ratio", type=float, default=0.5, help="Fraction of remaining images to VAL")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["symlink", "copy", "move", "none"], default="symlink")
    ap.add_argument("--pretty", action="store_true", default=False, help=argparse.SUPPRESS)  # keep simple
    args, unknown = ap.parse_known_args()
    # Backward-compat: allow --pretty without value
    args.pretty = ("--pretty" in sys.argv)

    ann_path, images_root, out_root = Path(args.ann), Path(args.images_root), Path(args.out)
    if not ann_path.exists(): raise SystemExit(f"[ERR] Annotations not found: {ann_path}")
    if not images_root.exists(): raise SystemExit(f"[ERR] Images root not found: {images_root}")

    coco = load_coco(ann_path)
    super2catids = build_super_indexes(coco["categories"])

    # Parse and resolve supercategories (robust to case/&/spaces)
    raw_supers = [s.strip() for s in args.classes.split(",") if s.strip()]
    ordered_supers: List[str] = []
    missing: List[Tuple[str, List[str]]] = []
    for s in raw_supers:
        if s in super2catids:
            ordered_supers.append(s)
            continue
        ns = norm(s)
        found = None
        for real in super2catids.keys():
            if norm(real) == ns:
                found = real
                break
        if found:
            ordered_supers.append(found)
        else:
            sugg = get_close_matches(s, list(super2catids.keys()), n=5, cutoff=0.6)
            missing.append((s, sugg))
    if missing:
        msg = ["\nUnknown supercategory names:"]
        for bad, sugg in missing:
            msg.append(f"  - '{bad}'" + (f" (did you mean: {', '.join(sugg)})" if sugg else ""))
        msg.append("\nAvailable supercategories:")
        for name in sorted(super2catids.keys()):
            msg.append(f"  - {name}")
        raise ValueError("\n".join(msg))

    # Allowed original categories (those belonging to chosen supercats)
    allowed_cat_ids: Set[int] = set()
    for s in ordered_supers:
        allowed_cat_ids.update(super2catids[s])

    # Select K-shot per supercat for TRAIN
    train_img_ids, train_ann_ids_per_img, remaining_img_ids, per_super_avail = pick_k_per_super(
        coco, ordered_supers, args.k, args.seed
    )

    # Split remaining to VAL/TEST
    val_ids, test_ids = split_remaining(remaining_img_ids, val_ratio=args.val_ratio, seed=args.seed)

    # Build subsets (still have original cat ids), then remap to supercats (1..N)
    train_coco = subset_coco(coco, train_img_ids, allowed_cat_ids, train_ann_filter=train_ann_ids_per_img)
    val_coco   = subset_coco(coco, val_ids,   allowed_cat_ids)
    test_coco  = subset_coco(coco, test_ids,  allowed_cat_ids)

    train_coco = remap_to_supercats(train_coco, coco, ordered_supers)
    val_coco   = remap_to_supercats(val_coco, coco, ordered_supers)
    test_coco  = remap_to_supercats(test_coco, coco, ordered_supers)

    # Write JSONs
    ann_dir = out_root / "annotations"
    ensure_dir(ann_dir)
    if args.pretty:
        ann_dir.joinpath("instances_train.json").write_text(json.dumps(train_coco, indent=2, ensure_ascii=False), encoding="utf-8")
        ann_dir.joinpath("instances_val.json").write_text(json.dumps(val_coco, indent=2, ensure_ascii=False), encoding="utf-8")
        ann_dir.joinpath("instances_test.json").write_text(json.dumps(test_coco, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        ann_dir.joinpath("instances_train.json").write_text(json.dumps(train_coco, ensure_ascii=False), encoding="utf-8")
        ann_dir.joinpath("instances_val.json").write_text(json.dumps(val_coco, ensure_ascii=False), encoding="utf-8")
        ann_dir.joinpath("instances_test.json").write_text(json.dumps(test_coco, ensure_ascii=False), encoding="utf-8")

    # Mirror images
    mirror_images("train", train_coco["images"], images_root, out_root, mode=args.mode)
    mirror_images("val",   val_coco["images"],   images_root, out_root, mode=args.mode)
    mirror_images("test",  test_coco["images"],  images_root, out_root, mode=args.mode)

    # Summary
    print("\n=== FEW-SHOT COCO (by SUPER-CATEGORY) ===")
    print("Order / new IDs:", ", ".join([f"{i+1}:{n}" for i, n in enumerate(ordered_supers)]))
    print(f"K per supercategory (train): {args.k}")
    for s, avail in per_super_avail.items():
        print(f"  - {s}: picked {min(args.k, avail)} / available {avail}")
    print(f"Train images: {len(train_coco['images'])}")
    print(f"Val images:   {len(val_coco['images'])}")
    print(f"Test images:  {len(test_coco['images'])}")
    print("Output root:", out_root)
    print("Annotations:", ann_dir)

if __name__ == "__main__":
    main()
