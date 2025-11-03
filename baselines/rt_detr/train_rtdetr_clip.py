import argparse, os, yaml
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# ---- OpenAI CLIP (NOT open_clip) ----
import clip  # pip install "git+https://github.com/openai/CLIP"


# -----------------------------
# Text-guided cosine head
# -----------------------------
class TextGuidedClsHead(nn.Module):
    """
    Cosine-similarity class head against CLIP text embeddings (one row per class).
    Plug-in replacement for RT-DETR dec/enc score heads.
    """
    def __init__(
        self,
        d_img: int,
        text_emb_clip: torch.Tensor,   # (C, d_clip) averaged per class
        clip_to_img_adapter: bool = True,
        scale_init: float = 10.0,
        class_gain: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        T = F.normalize(text_emb_clip.to(device).float(), dim=-1)  # (C, d_clip)
        self.register_buffer("text_emb_clip", T, persistent=False)
        d_clip = T.shape[-1]

        # Adapter d_clip -> d_img if needed
        if clip_to_img_adapter and d_clip != d_img:
            self.adapter = nn.Linear(d_clip, d_img, bias=False)
        elif d_clip == d_img:
            self.adapter = nn.Identity()
        else:
            # force adapter if dims mismatch
            self.adapter = nn.Linear(d_clip, d_img, bias=False)

        # learnable temperature
        self.logit_scale = nn.Parameter(torch.tensor(scale_init).log())

        # optional per-class gain
        self.use_class_gain = class_gain
        if class_gain:
            self.class_gain = nn.Parameter(torch.ones(T.shape[0]))

    def forward(self, decoder_feats: torch.Tensor) -> torch.Tensor:
        """
        decoder_feats: (B, Q, D) — decoder query features
        returns logits: (B, Q, C)
        """
        B, Q, D = decoder_feats.shape
        X = F.normalize(decoder_feats, dim=-1)      # (B,Q,D_img)

        T = self.text_emb_clip                      # (C,d_clip)
        T = self.adapter(T)                         # (C,D_img)
        T = F.normalize(T, dim=-1)                  # (C,D_img)

        logits = torch.einsum('bqd,cd->bqc', X, T)  # cosine similarity
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        logits = logits * scale
        if self.use_class_gain:
            logits = logits * self.class_gain.view(1, 1, -1)
        return logits


# -----------------------------
# Helpers
# -----------------------------
def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_class_names(data_yaml_path: str) -> List[str]:
    with open(data_yaml_path, "r") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        names = [name for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
    if not isinstance(names, list) or not all(isinstance(n, (str, int)) for n in names):
        raise ValueError("data.yaml must contain 'names' as list or id->name dict.")
    return [str(x) for x in names]

@torch.no_grad()
def averaged_text_embeddings(clip_model, tokenizer, class_names: List[str], device: str, templates: List[str]) -> torch.Tensor:
    vecs = []
    for cname in class_names:
        variants = [t.format(cname) for t in templates]
        toks = tokenizer(variants).to(device)
        T = clip_model.encode_text(toks).float()   # (K, d)
        T = F.normalize(T, dim=-1)
        T = T.mean(0, keepdim=True)               # (1, d)
        T = F.normalize(T, dim=-1)
        vecs.append(T)
    T_all = torch.cat(vecs, dim=0)                # (C, d)
    return T_all

def replace_heads(ultra_model, new_head: nn.Module, verbose=True):
    """Replace dec_score_head (+ enc_score_head if present). Fallback to common aliases."""
    found_dec = found_enc = False
    for name, module in ultra_model.model.named_modules():
        if hasattr(module, "dec_score_head") and not found_dec:
            module._old_dec_score_head = module.dec_score_head
            module.dec_score_head = new_head
            found_dec = True
            if verbose: print(f"[OK] Replaced {name}.dec_score_head")
        if hasattr(module, "enc_score_head") and not found_enc:
            module._old_enc_score_head = module.enc_score_head
            module.enc_score_head = new_head
            found_enc = True
            if verbose: print(f"[OK] Replaced {name}.enc_score_head")

    if not found_dec:
        # Fallback: try common aliases (implementation-dependent)
        for name, module in ultra_model.model.named_modules():
            for attr in ("cls", "cls_head", "classifier", "score_head"):
                if hasattr(module, attr):
                    setattr(module, f"_old_{attr}", getattr(module, attr))
                    setattr(module, attr, new_head)
                    found_dec = True
                    if verbose: print(f"[OK] Replaced {name}.{attr} (fallback)")
                    break
            if found_dec:
                break

    if not found_dec:
        raise RuntimeError("Could not find a classification/score head to replace.")

    return found_dec, found_enc

def unfreeze_for_finetune(ultra_model, text_head: nn.Module, verbose=True, freeze_backbone_layers: int = 0):
    """
    Prepare model for few-shot finetuning:
      - freeze all params
      - unfreeze bbox & score heads (enc/dec) and the last decoder block
      - ensure text_head params are trainable
      - optionally re-freeze the first `freeze_backbone_layers` children of the backbone
    """
    # Freeze all first
    for p in ultra_model.model.parameters():
        p.requires_grad = False

    # Unfreeze bbox heads + score heads (both enc/dec)
    touched = 0
    for name, m in ultra_model.model.named_modules():
        lname = name.lower()
        if any(k in lname for k in ["dec_bbox_head", "enc_bbox_head", "bbox_head"]):
            for p in m.parameters():
                p.requires_grad = True
            touched += 1
        if any(k in lname for k in ["dec_score_head", "enc_score_head", "cls", "classifier", "score_head", "cls_head"]):
            for p in m.parameters():
                p.requires_grad = True
            touched += 1

    # Unfreeze the **last** decoder block (helps alignment)
    last_decoder = None
    for name, m in ultra_model.model.named_modules():
        if "decoder" in name.lower():
            last_decoder = m
    if last_decoder is not None:
        for p in last_decoder.parameters():
            p.requires_grad = True
        if verbose: print(f"[OK] Unfroze last decoder block: {last_decoder.__class__.__name__}")

    # Ensure text head is trainable (adapter + scale)
    for p in text_head.parameters():
        p.requires_grad = True

    # Optionally freeze first N backbone child modules
    if freeze_backbone_layers and int(freeze_backbone_layers) > 0:
        n = int(freeze_backbone_layers)
        frozen = 0
        # Heuristic: find modules whose name contains 'backbone' (or common aliases)
        candidates = [(name, m) for name, m in ultra_model.model.named_modules()
                      if "backbone" in name.lower() or "stem" in name.lower()]
        if not candidates:
            # fallback: look for classes that look like backbone
            candidates = [(name, m) for name, m in ultra_model.model.named_modules()
                          if "backbone" in m.__class__.__name__.lower() or "stem" in m.__class__.__name__.lower()]

        for name, bm in candidates:
            children = list(bm.children())
            if not children:
                continue
            for child in children:
                if frozen >= n:
                    break
                for p in child.parameters():
                    p.requires_grad = False
                frozen += 1
                if verbose:
                    print(f"[OK] Froze backbone child #{frozen} in {name} -> {child.__class__.__name__}")
            if frozen >= n:
                break

        if verbose:
            if frozen == 0:
                print(f"[WARN] Requested freezing {n} backbone children but found none.")
            else:
                print(f"[INFO] Frozen {frozen}/{n} backbone child modules.")

    if verbose:
        learnable = sum(p.numel() for p in ultra_model.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in ultra_model.model.parameters())
        print(f"[INFO] Trainable params: {learnable}/{total} ({100.0*learnable/total:.2f}%) ; finetune modules touched: {touched}")

def set_nc_and_names(ultra_model, class_names: List[str]):
    C = len(class_names)
    if hasattr(ultra_model.model, "nc"):
        ultra_model.model.nc = C

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to YAML config")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    # ---- Required blocks ----
    model_path = cfg["model"]
    data_yaml = cfg["data"]
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load YOLO/RT-DETR ----
    model = YOLO(model_path)
    model.to(device)

    # ---- Classes ----
    class_names = read_class_names(data_yaml)
    print(f"[INFO] Classes ({len(class_names)}): {class_names}")
    set_nc_and_names(model, class_names)

    # ---- CLIP (OpenAI) ----
    clip_name = cfg.get("clip", {}).get("model", "ViT-B/32")
    print(f"[INFO] Loading CLIP: {clip_name}")
    clip_model, _ = clip.load(clip_name, device=device)

    # ---- Prompt templates ----
    templates = cfg.get("text_head", {}).get("templates", [
        "a photo of a {}.",
        "an image of a {}.",
        "a {} on a table.",
        "a close-up photo of a {}.",
        "a cropped photo of a {}."
    ])

    # ---- Averaged text embeddings (C, d_clip) ----
    with torch.no_grad():
        T = averaged_text_embeddings(clip_model, clip.tokenize, class_names, device, templates)

    # ---- Decoder feature dim (assume 256; RT-DETR default). If your build differs, adjust here. ----
    decoder_dim = cfg.get("model_dim", 256)

    # ---- Build text head ----
    th_cfg = cfg.get("text_head", {})
    text_head = TextGuidedClsHead(
        d_img=decoder_dim,
        text_emb_clip=T,
        clip_to_img_adapter=th_cfg.get("adapter_learnable", True) or True,  # if dims differ, we adapt
        scale_init=th_cfg.get("scale_init", 10.0),
        class_gain=th_cfg.get("class_gain", True),
        device=device,
    )

    # ---- Replace heads ----
    found_dec, found_enc = replace_heads(model, text_head, verbose=True)

    # ---- Unfreeze/right parts + optional backbone-layer freezing ----
    # cfg may specify number of backbone children to freeze via 'freeze_backbone' or plain 'freeze'
    freeze_layers = None
    if "freeze_backbone" in cfg:
        freeze_layers = int(cfg.get("freeze_backbone") or 0)
    elif "freeze" in cfg:
        # interpret legacy 'freeze' as backbone-layer count if present (common UX)
        try:
            freeze_layers = int(cfg.get("freeze") or 0)
        except Exception:
            freeze_layers = 0

    unfreeze_for_finetune(model, text_head, verbose=True, freeze_backbone_layers=freeze_layers or 0)

    # ---- Sanity prints ----
    print(f"[INFO] logit_scale (effective): {text_head.logit_scale.exp().item():.3f}")
    print(f"[INFO] text_emb_clip shape: {tuple(text_head.text_emb_clip.shape)}")
    if hasattr(text_head, "adapter") and hasattr(text_head.adapter, "weight"):
        ws = tuple(text_head.adapter.weight.shape)
        print(f"[INFO] adapter weight shape: {ws}")

    # ---- Train args from cfg ----
    train_args = dict(
        data=data_yaml,
        epochs=cfg.get("epochs"),
        imgsz=cfg.get("imgsz"),
        batch=cfg.get("batch"),
        optimizer=cfg.get("optimizer"),
        lr0=cfg.get("lr0"),
        lrf=cfg.get("lrf"),
        weight_decay=cfg.get("weight_decay"),
        cos_lr=cfg.get("cos_lr"),
        warmup_epochs=cfg.get("warmup_epochs"),
        amp=cfg.get("amp"),
        mosaic=cfg.get("mosaic"),
        mixup=cfg.get("mixup"),
        copy_paste=cfg.get("copy_paste"),
        close_mosaic=cfg.get("close_mosaic"),
        seed=cfg.get("seed"),
        project=cfg.get("project"),
        name=cfg.get("name"),
        rect=cfg.get("rect"),
        plots=cfg.get("plots"),
        verbose=cfg.get("verbose")
    )

    print("[INFO] Starting training ...")
    model.train(**train_args)
    print("[INFO] Done. Best run in:",
          os.path.join(model.overrides.get("project", "runs"),
                       model.overrides.get("name", "")))


if __name__ == "__main__":
    main()
