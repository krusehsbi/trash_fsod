#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot RT-DETR training with CLIP-guided logit-fusion head.
Single argument: --config path/to/train.yaml

Config YAML fields (example)
-------------------------------------------------
model: rtdetr-l.pt
data: data/coco.yaml
device: cuda
epochs: 50
batch: 16
imgsz: 640
workers: 8
lr0: 0.001
weight_decay: 5e-5
project: runs-ovd
name: rtdetr_clip_zs
patience: 50
plots: false
verbose: false

clip:
  model: "ViT-L/14@336px"

text_head:
  templates:
    - "a photo of a {}"
    - "an image of a {}"
    - "a close-up of a {}"
  fusion: "amax"        # or "logsumexp"
  center_text: true
-------------------------------------------------
"""
import argparse, os, yaml, warnings
from typing import Sequence, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# OpenAI CLIP
import clip  # pip install "git+https://github.com/openai/CLIP"


# -----------------------------
# Utilities
# -----------------------------
def read_class_names(data_yaml_path: str) -> List[str]:
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if not isinstance(names, list) or not names:
        raise ValueError("Could not read class names from data yaml (expected 'names' list).")
    return names


def ensure_templates_list(obj) -> List[str]:
    if obj is None:
        return ["a photo of a {}", "an image of a {}", "a close-up of a {}"]
    if isinstance(obj, str):
        parts = [p.strip() for p in obj.split(",") if p.strip()]
        arr = parts if parts else [obj.strip()]
    elif isinstance(obj, list):
        arr = obj
    else:
        raise ValueError("text_head.templates must be list or comma-separated string")
    fixed = []
    for t in arr:
        fixed.append(t if "{}" in t else (t + " {}"))
    return fixed


# -----------------------------
# Text encoding helpers (fp32)
# -----------------------------
@torch.no_grad()
def stacked_text_embeddings(
    clip_model,
    tokenizer,
    class_names: Sequence[str],
    templates: Sequence[str],
    device: torch.device,
    normalize_per_template: bool = True,
) -> torch.Tensor:
    """
    Returns per-template text embeddings: (C, K, D_clip).
    Keeps CLIP text in eval + float32.
    """
    clip_was_training = clip_model.training
    clip_model.eval()
    all_T = []
    for cname in class_names:
        prompts = [t.format(cname) for t in templates]
        toks = tokenizer(prompts).to(device)
        # Force fp32 for text
        with torch.amp.autocast("cuda", enabled=False):
            T = clip_model.encode_text(toks).float()  # (K, D_clip)
        if normalize_per_template:
            T = F.normalize(T, dim=-1)
        all_T.append(T.unsqueeze(0))
    T = torch.cat(all_T, dim=0)  # (C, K, D_clip)
    if clip_was_training:
        clip_model.train()
    return T


def compute_text_center(T: torch.Tensor) -> torch.Tensor:
    """Compute global center over (C,K,D). Returns (1,1,D)."""
    CK, D = (T.shape[0] * T.shape[1], T.shape[2])
    center = T.reshape(CK, D).mean(0, keepdim=True)
    return center.view(1, 1, D)


# -----------------------------
# Logit-fusion cosine head
# -----------------------------
class TextGuidedClsHeadLogitFusion(nn.Module):
    """
    Zero-shot classification head with:
      - Adapter mapping CLIP text space -> detector space
      - Logit-space fusion over K templates per class (amax or logsumexp)
      - Learnable temperature (logit_scale) initialized near ln(100)
      - Optional per-class gain
      - Background logit ("none of the above")
      - Optional centering / debias of text embeddings before final norm
    Expected input:
      decoder_feats: (B, Q, D_img)
    Returns:
      logits over C classes (B,Q,C) or (B,Q,C+1) if return_with_bg=True.
    """
    def __init__(
        self,
        text_emb_clip_stacked: torch.Tensor,  # (C,K,D_clip)
        d_img: int,
        fusion: str = "amax",  # "amax" or "logsumexp"
        class_gain: bool = True,
        center_text: bool = True,
        init_logit_scale: float = 4.6052,  # ln(100)
        return_with_bg: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert fusion in ("amax", "logsumexp")
        self.fusion = fusion
        self.center_text = center_text
        self.return_with_bg = return_with_bg

        T = text_emb_clip_stacked
        if device is not None:
            T = T.to(device)
        self.register_buffer("text_emb_clip", T.float(), persistent=False)  # (C,K,Dc)
        self.C, self.K, self.Dc = T.shape

        # Adapter: CLIP dim -> detector dim
        self.adapter = nn.Linear(self.Dc, d_img, bias=False)

        # Temperature
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale, dtype=torch.float32))

        # Optional per-class gain
        self.use_class_gain = class_gain
        if class_gain:
            self.class_gain = nn.Parameter(torch.ones(self.C, dtype=torch.float32))

        # Background logit
        self.bg_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Text center for debiasing
        if center_text:
            with torch.no_grad():
                mu = compute_text_center(self.text_emb_clip)  # (1,1,Dc)
            self.register_buffer("text_center", mu, persistent=False)
        else:
            self.register_buffer("text_center", torch.zeros(1, 1, self.Dc), persistent=False)

    def _fuse_over_templates(self, sim: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
        # sim: (B,Q,C,K) already scaled
        if self.fusion == "amax":
            return sim.amax(dim=-1)  # (B,Q,C)
        else:
            # log-sum-exp with temperature tau
            return torch.logsumexp(sim / tau, dim=-1) * tau

    def forward(self, decoder_feats: torch.Tensor) -> torch.Tensor:
        """
        decoder_feats: (B, Q, D_img)
        returns: logits (B, Q, C) or (B, Q, C+1) if return_with_bg
        """
        B, Q, D = decoder_feats.shape
        # Normalize decoder features
        X = F.normalize(decoder_feats, dim=-1)  # (B,Q,D_img)

        # Prepare text bank in detector space
        T = self.text_emb_clip  # (C,K,Dc)
        # Center (debias) then adapt and normalize per-template
        if self.center_text:
            T = T - self.text_center
        T = self.adapter(T.view(self.C * self.K, self.Dc))  # (C*K, D_img)
        T = F.normalize(T, dim=-1).view(self.C, self.K, D)  # (C,K,D_img)

        # Cosine similarity to each template
        # sim: (B,Q,C,K)
        sim = torch.einsum("bqd,ckd->bqck", X, T)

        # Scale
        scale = self.logit_scale.exp().clamp_(1.0, 100.0)
        sim = sim * scale

        # Fuse over templates K -> logits (B,Q,C)
        logits = self._fuse_over_templates(sim)  # (B,Q,C)

        # Per-class gain (broadcast)
        if self.use_class_gain:
            logits = logits * self.class_gain.view(1, 1, self.C)

        if self.return_with_bg:
            bg = self.bg_logit.expand(B, Q, 1)
            logits = torch.cat([logits, bg], dim=-1)  # (B,Q,C+1)

        return logits


# -----------------------------
# RT-DETR head shim + hooks
# -----------------------------
class ClassifierShim(nn.Module):
    """
    Wraps an existing detection head that outputs (bbox/objectness/aux...)
    and replaces/augments its classification logits with our
    TextGuidedClsHeadLogitFusion computed from decoder/query features.

    IMPORTANT: We expect the decoder/query features to be exposed on the
    base detection head (or the DetectionModel) as attribute 'decoder_out'
    with shape (B,Q,D). The helper install_decoder_capture() installs a
    forward hook to populate this attribute during the forward pass.
    """
    def __init__(self, base_head: nn.Module, zs_head: TextGuidedClsHeadLogitFusion, feat_hook_name: str = "decoder_out"):
        super().__init__()
        self.base_head = base_head
        self.zs_head = zs_head
        self.feat_hook_name = feat_hook_name

    def forward(self, x):
        out = self.base_head(x)  # run original detect head
        decoder_feats = getattr(self.base_head, self.feat_hook_name, None)
        if decoder_feats is None:
            decoder_feats = getattr(self, self.feat_hook_name, None)
        if decoder_feats is None:
            warnings.warn("Decoder/query features not found; ensure install_decoder_capture() is called.")
            return out
        # zs logits (B,Q,C [+1 bg]) are computed and attached for downstream use
        zs_logits = self.zs_head(decoder_feats)
        setattr(self.base_head, "zs_logits", zs_logits)
        return out


def _is_detect_head(mod: nn.Module) -> bool:
    n = mod.__class__.__name__.lower()
    return any(k in n for k in ["detect", "head", "rtdetr"])


def install_decoder_capture(det_model: nn.Module, feat_attr: str = "decoder_out") -> bool:
    """
    Install a forward hook on a module likely to output decoder/query features.
    We search for modules with names containing 'decoder' or 'transformer'.
    Captured tensor is stored on det_model.<feat_attr> with shape (B,Q,D).
    """
    target = None
    for name, m in det_model.named_modules():
        lname = name.lower()
        if any(k in lname for k in ["decoder", "transformer"]):
            target = m  # last match wins
    if target is None:
        warnings.warn("Could not find a 'decoder'/'transformer' module to hook; "
                      "please point install_decoder_capture() to the right module.")
        return False

    def _hook(module, inputs, output):
        val = output
        # If tuple/list, find a 3D tensor (B,Q,D)
        if isinstance(val, (tuple, list)):
            pick = None
            for v in reversed(val):
                if torch.is_tensor(v) and v.dim() == 3:
                    pick = v
                    break
            val = pick if pick is not None else val[0]
        if torch.is_tensor(val):
            if val.dim() == 3:
                setattr(det_model, feat_attr, val)
            elif val.dim() > 3:
                setattr(det_model, feat_attr, val.flatten(2))  # (B,*,D)
        # else leave unset; shim will warn

    target.register_forward_hook(_hook)
    return True


def attach_head_to_rtdetr(yolo: YOLO, zs_head: TextGuidedClsHeadLogitFusion, feat_attr: str = "decoder_out") -> None:
    """
    Robustly attach the zero-shot shim to an Ultralytics DetectionModel.
    - Prefer replacing the last module in the internal container (det.model / det.model.model)
    - Else, scan children to locate a detect/head-like module
    """
    det = yolo.model  # DetectionModel
    container = getattr(det, "model", None)   # many Ultralytics models keep layers here

    # Case A: container is a Sequential/ModuleList -> replace last
    if isinstance(container, (nn.Sequential, nn.ModuleList)) and len(container) > 0:
        base_head = container[-1]
        shim = ClassifierShim(base_head, zs_head, feat_hook_name=feat_attr)
        container[-1] = shim
        return

    # Case B: find a likely head among direct children
    for name, child in det.named_children():
        if _is_detect_head(child):
            shim = ClassifierShim(child, zs_head, feat_hook_name=feat_attr)
            setattr(det, name, shim)
            return

    # Case C: scan deepest modules; replace last detect-like
    last_name, last_mod = None, None
    for name, m in det.named_modules():
        if _is_detect_head(m):
            last_name, last_mod = name, m
    if last_mod is not None:
        parent = det
        parts = last_name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        shim = ClassifierShim(last_mod, zs_head, feat_hook_name=feat_attr)
        setattr(parent, parts[-1], shim)
        return

    raise RuntimeError("Could not locate a detection head to wrap; adjust attach_head_to_rtdetr().")


def auto_detect_decoder_dim(yolo: YOLO, fallback: int = 256) -> int:
    """
    Best-effort attempt to find decoder feature dim. Falls back to 256.
    """
    det = yolo.model
    for m in det.modules():
        for attr in ("c2", "c3", "ch", "dim", "d_model", "embed_dim", "hidden_dim"):
            if hasattr(m, attr):
                val = getattr(m, attr)
                if isinstance(val, int) and 64 <= val <= 1024:
                    return val
    return fallback


def unfreeze_for_base_training(yolo: YOLO):
    """
    Unfreeze only:
      - our shim + zs head
      - the LAST decoder/transformer block (best-effort)
    Keep backbone/neck frozen.
    """
    det = yolo.model
    for p in det.parameters():
        p.requires_grad = False

    # last decoder/transformer block
    last_decoder = None
    for name, module in reversed(list(det.named_modules())):
        if any(k in name.lower() for k in ["decoder", "transformer"]):
            last_decoder = module
            break
    if last_decoder:
        for p in last_decoder.parameters():
            p.requires_grad = True

    # shim + zs head (detect head replacement)
    for _, m in det.named_modules():
        if isinstance(m, ClassifierShim):
            for p in m.parameters():
                p.requires_grad = True
            break


# -----------------------------
# Train pipeline (config-only)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config file")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Basic params
    model_path = cfg.get("model")
    data_yaml = cfg.get("data")
    if not model_path or not data_yaml:
        raise ValueError("Config must include 'model' and 'data'.")

    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load classes
    class_names = read_class_names(data_yaml)
    print(f"[INFO] Classes ({len(class_names)}): {class_names}")

    # Load model
    model = YOLO(model_path)
    model.to(device)

    # Load CLIP
    clip_name = (cfg.get("clip") or {}).get("model", "ViT-L/14@336px")
    print(f"[INFO] Loading CLIP: {clip_name}")
    clip_model, _ = clip.load(clip_name, device=device)

    # Templates & text head cfg
    th_cfg = cfg.get("text_head") or {}
    templates = ensure_templates_list(th_cfg.get("templates"))
    fusion = th_cfg.get("fusion", "amax")
    center_text = bool(th_cfg.get("center_text", True))

    print(f"[INFO] Templates ({len(templates)}): {templates}")
    print(f"[INFO] Fusion: {fusion} | Center text: {center_text}")

    # Build per-template text embeddings
    T_stack = stacked_text_embeddings(
        clip_model=clip_model,
        tokenizer=clip.tokenize,
        class_names=class_names,
        templates=templates,
        device=device,
        normalize_per_template=True,
    )

    # Find decoder feature dim
    D_img = auto_detect_decoder_dim(model, fallback=256)
    print(f"[INFO] Using decoder feature dim: {D_img}")

    # Build ZS head
    zs_head = TextGuidedClsHeadLogitFusion(
        text_emb_clip_stacked=T_stack,
        d_img=D_img,
        fusion=fusion,
        center_text=center_text,
        class_gain=True,
        return_with_bg=True,
        device=torch.device(device),
    )

    # Install decoder capture first (so the shim will find features)
    install_decoder_capture(model.model, feat_attr="decoder_out")

    # Attach and unfreeze
    attach_head_to_rtdetr(model, zs_head, feat_attr="decoder_out")
    unfreeze_for_base_training(model)

    # Train args from cfg (pass-through to Ultralytics)
    train_args = dict(
        data=data_yaml,
        epochs=cfg.get("epochs", 50),
        batch=cfg.get("batch", 16),
        imgsz=cfg.get("imgsz", 640),
        workers=cfg.get("workers", 8),
        lr0=cfg.get("lr0", 1e-3),
        weight_decay=cfg.get("weight_decay", 5e-5),
        project=cfg.get("project", "runs-ovd"),
        name=cfg.get("name", "rtdetr_clip_zs"),
        device=device,
        plots=cfg.get("plots", False),
        verbose=cfg.get("verbose", False),
        patience=cfg.get("patience", 50),
    )
    print("[INFO] Training with args:", train_args)
    model.train(**train_args)
    print("[INFO] Done. Best run in:",
          os.path.join(model.overrides.get("project", "runs"),
                       model.overrides.get("name", "")))


if __name__ == "__main__":
    main()
