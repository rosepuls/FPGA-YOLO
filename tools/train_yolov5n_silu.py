from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "yolov5n_gray_silu_baseline.yaml"
REQUIRED_KEYS = {
    "data",
    "cfg",
    "weights",
    "hyp",
    "imgsz",
    "channels",
    "epochs",
    "batch_size",
    "seed",
    "device",
    "workers",
    "cache",
    "project",
    "name",
    "exist_ok",
}


def load_run_config(path: Path) -> dict:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("run config must be a YAML mapping")

    missing = sorted(REQUIRED_KEYS - config.keys())
    if missing:
        raise ValueError(f"run config missing keys: {', '.join(missing)}")
    if config["channels"] != 1:
        raise ValueError("baseline channels must be 1")
    if config["imgsz"] != 256:
        raise ValueError("baseline imgsz must be 256")
    if config["cache"] is not None:
        raise ValueError("baseline image cache must be disabled")

    data_path = PROJECT_ROOT / config["data"]
    with data_path.open(encoding="utf-8") as stream:
        data_config = yaml.safe_load(stream)
    if data_config.get("nc") != 10:
        raise ValueError("dataset nc must be 10")

    for key in ("data", "cfg", "hyp"):
        if not (PROJECT_ROOT / config[key]).is_file():
            raise FileNotFoundError(PROJECT_ROOT / config[key])
    return config


def _resolve_existing_path(project_root: Path, value: str) -> Path | None:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve() if candidate.exists() else None


def _checkpoint_state_dict(project_root: Path, weights: str) -> dict[str, torch.Tensor]:
    from utils.torch_utils import load_checkpoint

    local_weights = _resolve_existing_path(project_root, weights)
    if local_weights is None:
        from utils.downloads import attempt_download

        local_weights = Path(attempt_download(weights)).resolve()
    checkpoint = load_checkpoint(local_weights, map_location="cpu")
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model" in checkpoint:
        return checkpoint["model"].float().state_dict()
    raise ValueError(f"unsupported checkpoint structure: {local_weights}")


def run_preflight(project_root: Path, run_config: dict) -> dict:
    project_root = Path(project_root).resolve()
    vendor_root = project_root / "third_party" / "yolov5"
    os.environ.setdefault(
        "YOLOV5_CONFIG_DIR",
        str(Path(tempfile.gettempdir()) / "fpga-yolo-ultralytics"),
    )
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "fpga-yolo-matplotlib"),
    )
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))

    from models.yolo import Model
    from utils.dataloaders import create_dataloader
    from utils.torch_utils import adapt_input_conv

    data_path = (project_root / run_config["data"]).resolve()
    model_path = (project_root / run_config["cfg"]).resolve()
    hyp_path = (project_root / run_config["hyp"]).resolve()
    with data_path.open(encoding="utf-8") as stream:
        data_config = yaml.safe_load(stream)
    with hyp_path.open(encoding="utf-8") as stream:
        hyp = yaml.safe_load(stream)

    dataset_root = Path(data_config["path"])
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    train_path = Path(data_config["train"])
    if not train_path.is_absolute():
        train_path = dataset_root / train_path

    model = Model(str(model_path), ch=run_config["channels"], nc=data_config["nc"])
    loader, _ = create_dataloader(
        str(train_path),
        imgsz=run_config["imgsz"],
        batch_size=2,
        stride=int(model.stride.max()),
        hyp=hyp,
        augment=False,
        workers=0,
        channels=run_config["channels"],
    )
    images = next(iter(loader))[0]

    source_state = _checkpoint_state_dict(project_root, run_config["weights"])
    adapted_state, adapted_key = adapt_input_conv(
        source_state,
        model.state_dict(),
        run_config["channels"],
    )
    if adapted_key is None:
        raise ValueError("baseline checkpoint must contain an RGB input convolution")
    expected_first_conv = source_state[adapted_key].mean(dim=1, keepdim=True)
    model.load_state_dict({adapted_key: adapted_state[adapted_key]}, strict=False)
    converted = torch.equal(model.state_dict()[adapted_key], expected_first_conv)

    model.eval()
    with torch.no_grad():
        model(images.float() / 255.0)

    report = {
        "batch_shape": list(images.shape),
        "first_conv_in_channels": model.model[0].conv.in_channels,
        "head_out_channels": [layer.out_channels for layer in model.model[-1].m],
        "pretrained_first_conv_key": adapted_key,
        "pretrained_first_conv_verified": converted,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    expected_shape = [1, run_config["imgsz"], run_config["imgsz"]]
    if report["batch_shape"][1:] != expected_shape:
        raise ValueError(f"unexpected batch shape: {report['batch_shape']}")
    if report["first_conv_in_channels"] != 1:
        raise ValueError("model first convolution is not single-channel")
    if report["head_out_channels"] != [45, 45, 45]:
        raise ValueError(f"unexpected detection heads: {report['head_out_channels']}")
    if not report["pretrained_first_conv_verified"]:
        raise ValueError("RGB-to-grayscale pretrained conversion verification failed")
    report["status"] = "ready"
    return report


def _training_kwargs(project_root: Path, run_config: dict) -> dict:
    arguments = dict(run_config)
    for key in ("data", "cfg", "hyp", "project"):
        arguments[key] = str((project_root / arguments[key]).resolve())
    local_weights = _resolve_existing_path(project_root, arguments["weights"])
    if local_weights is not None:
        arguments["weights"] = str(local_weights)
    return arguments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the single-channel YOLOv5n SiLU baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args(argv)

    run_config = load_run_config(arguments.config)
    report = run_preflight(PROJECT_ROOT, run_config)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if arguments.preflight_only:
        return 0

    import train

    train.run(**_training_kwargs(PROJECT_ROOT, run_config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
