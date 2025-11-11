import argparse
import yaml
from pathlib import Path
from ultralytics import YOLOWorld

def main():
    # 1) Read training config path from CLI
    parser = argparse.ArgumentParser(description="Train YOLOWorld from a full YAML config (incl. augmentations).")
    parser.add_argument("--config", type=str, help="Path to training config YAML (e.g., 30shot_train.yaml)")
    args = parser.parse_args()

    # 2) Load full config (everything comes from here)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Training config must parse to a dict.")

    # 3) Required fields
    if "model" not in cfg:
        raise ValueError("Missing required key: 'model' in training config.")
    if "data" not in cfg:
        raise ValueError("Missing required key: 'data' in training config.")

    # 4) Init model
    model_path = cfg["model"]
    model = YOLOWorld(model_path)

    # 5) Load dataset.yaml to set open-vocabulary classes automatically
    with open(cfg["data"], "r") as f:
        dataset_yaml = yaml.safe_load(f) or {}
    names = dataset_yaml.get("names")
    if names is None:
        raise ValueError("Dataset YAML must contain a 'names' field.")
    class_names = list(names.values()) if isinstance(names, dict) else names
    model.set_classes(class_names)

    # 6) Train with ALL kwargs from config (including augmentations)
    #    Remove non-train sections like a nested 'val' block; everything else is passed through.
    train_kwargs = dict(cfg)
    train_kwargs.pop("model", None)   # already used
    val_kwargs = train_kwargs.pop("val", None)  # optional separate val section

    print("Starting training with config keys:", sorted(train_kwargs.keys()))
    model.train(**train_kwargs)

    # 7) Optional validation using a dedicated 'val' section if provided
    if isinstance(val_kwargs, dict) and len(val_kwargs):
        print("Running validation with config keys:", sorted(val_kwargs.keys()))
        model.val(**val_kwargs)

if __name__ == "__main__":
    main()
