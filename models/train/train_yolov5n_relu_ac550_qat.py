"""Fine-tune the grayscale YOLOv5n-ReLU checkpoint with AC550 QAT.

This entry intentionally delegates data loading, loss computation, checkpointing,
and validation to the vendored YOLOv5 training loop.  A small pre-optimizer hook
in that loop prepares the model and forces FP32 rather than AMP arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELU_ROOT = PROJECT_ROOT / "third_party" / "yolov5-relu"
DEFAULT_HYP = PROJECT_ROOT / "configs" / "hyp.ac550_qat.yaml"


QatPhase = Literal["collect", "weight", "train", "freeze"]


def validate_qat_schedule(
    *, observer_epochs: int, activation_quant_epoch: int, freeze_observer_epoch: int, freeze_bn_epoch: int
) -> None:
    if observer_epochs < 0:
        raise ValueError("qat-observer-epochs must be non-negative")
    if activation_quant_epoch < observer_epochs:
        raise ValueError("qat-activation-epoch must be at least qat-observer-epochs")
    if freeze_observer_epoch < activation_quant_epoch:
        raise ValueError("qat-freeze-observer-epoch must be at least qat-activation-epoch")
    if freeze_bn_epoch < freeze_observer_epoch:
        raise ValueError("qat-freeze-bn-epoch must be at least qat-freeze-observer-epoch")


def qat_phase_for_epoch(
    epoch: int, *, observer_epochs: int, activation_quant_epoch: int, freeze_observer_epoch: int
) -> QatPhase:
    if epoch < observer_epochs:
        return "collect"
    if epoch < activation_quant_epoch:
        return "weight"
    if epoch < freeze_observer_epoch:
        return "train"
    return "freeze"


def should_freeze_batch_norm(epoch: int, *, freeze_bn_epoch: int) -> bool:
    return epoch >= freeze_bn_epoch


def is_deployment_candidate_epoch(epoch: int, *, freeze_observer_epoch: int, freeze_bn_epoch: int) -> bool:
    """Only checkpoints with frozen scales and frozen BN statistics may be exported."""
    return epoch >= max(freeze_observer_epoch, freeze_bn_epoch)


def scalar_fitness(value: object) -> float:
    """Normalize YOLOv5's scalar-or-one-element-array fitness callback value."""
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def copy_final_qat_checkpoint(
    save_dir: Path,
    *,
    completed_epoch: int,
    deployment_candidate_start_epoch: int,
    source_name: str = "last.pt",
) -> Path:
    """Publish the final frozen-QAT checkpoint after the vendor training loop returns.

    Checkpoint publication used to run in the ``on_model_save`` callback.  The
    first eligible callback is the BN-freeze boundary, so a filesystem error
    there could abort the whole training run at epoch 20.  Keep the training
    loop free of deployment file operations and publish only after it exits.
    """
    if completed_epoch < deployment_candidate_start_epoch:
        raise RuntimeError(
            "QAT ended before observer and batch-normalization statistics were frozen; "
            "no deployable checkpoint was produced"
        )

    source = save_dir / "weights" / source_name
    if not source.is_file():
        raise FileNotFoundError(f"QAT training completed without checkpoint: {source}")
    destination = source.with_name("best_qat.pt")
    temporary = destination.with_suffix(".pt.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _import_vendor_training():
    """Load the vendored package first, while retaining ``models.qat`` on its path."""
    if str(RELU_ROOT) not in sys.path:
        sys.path.insert(0, str(RELU_ROOT))
    import models

    vendor_models = str(RELU_ROOT / "models")
    if vendor_models not in models.__path__:
        models.__path__.append(vendor_models)
    project_models = str(PROJECT_ROOT / "models")
    if project_models not in models.__path__:
        models.__path__.append(project_models)
    import train as vendor_train
    from utils.callbacks import Callbacks
    from models.qat.ac550_qat_graph import prepare_ac550_qat, qat_metadata, set_qat_phase

    return vendor_train, Callbacks, prepare_ac550_qat, qat_metadata, set_qat_phase


def _freeze_batch_norm(model) -> None:
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AC550 signed-INT8 QAT fine-tuning for YOLOv5n-ReLU grayscale.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--hyp", type=Path, default=DEFAULT_HYP)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--channels", type=int, choices=(1,), default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", choices=("SGD", "Adam", "AdamW"), default="SGD")
    parser.add_argument("--noplots", action="store_true", help="disable YOLO plot threads during QAT training")
    parser.add_argument("--qat-observer-epochs", type=int, default=5)
    parser.add_argument("--qat-activation-epoch", type=int, default=15)
    parser.add_argument("--qat-freeze-observer-epoch", type=int, default=45)
    parser.add_argument("--qat-freeze-bn-epoch", type=int, default=50)
    parser.add_argument("--qat-observer-momentum", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--name", default="yolov5n_gray_relu_256_qat_v1")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def run(arguments: argparse.Namespace) -> Path:
    validate_qat_schedule(
        observer_epochs=arguments.qat_observer_epochs,
        activation_quant_epoch=arguments.qat_activation_epoch,
        freeze_observer_epoch=arguments.qat_freeze_observer_epoch,
        freeze_bn_epoch=arguments.qat_freeze_bn_epoch,
    )
    if not 0.0 <= arguments.qat_observer_momentum < 1.0:
        raise ValueError("qat-observer-momentum must be in [0, 1)")
    if not arguments.weights.is_file() or not arguments.data.is_file() or not arguments.hyp.is_file():
        raise FileNotFoundError("weights, data, and hyp must each name an existing file")

    vendor_train, callbacks_type, prepare_ac550_qat, qat_metadata, set_qat_phase = _import_vendor_training()
    from utils.general import strip_optimizer
    options = vendor_train.parse_opt(True)
    options.weights = str(arguments.weights.resolve())
    options.data = str(arguments.data.resolve())
    options.hyp = str(arguments.hyp.resolve())
    options.imgsz = arguments.imgsz
    options.channels = arguments.channels
    options.epochs = arguments.epochs
    options.batch_size = arguments.batch_size
    options.device = arguments.device
    options.workers = arguments.workers
    options.optimizer = arguments.optimizer
    options.noplots = arguments.noplots
    options.seed = arguments.seed
    options.project = str(arguments.project.resolve())
    options.name = arguments.name
    options.exist_ok = arguments.exist_ok
    options.qat_disable_amp = True

    state: dict[str, object] = {
        "epoch": 0,
        "last_completed_epoch": -1,
        "best_deployment_fitness": float("-inf"),
    }
    callbacks = callbacks_type()

    def write_metadata(status: str) -> None:
        model = state.get("model")
        if model is None:
            return
        metadata = {
            **qat_metadata(model),
            "status": status,
            "schedule": {
                "observer_epochs": arguments.qat_observer_epochs,
                "activation_quant_epoch": arguments.qat_activation_epoch,
                "freeze_observer_epoch": arguments.qat_freeze_observer_epoch,
                "freeze_bn_epoch": arguments.qat_freeze_bn_epoch,
                "deployment_candidate_start_epoch": max(
                    arguments.qat_freeze_observer_epoch, arguments.qat_freeze_bn_epoch
                ),
            },
            "amp_enabled": False,
            "seed": arguments.seed,
            "source_checkpoint_sha256": _sha256(arguments.weights.resolve()),
            "data_yaml_sha256": _sha256(arguments.data.resolve()),
            "hyp_yaml_sha256": _sha256(arguments.hyp.resolve()),
        }
        if "deployment_checkpoint" in state:
            metadata["deployment_candidate"] = {
                "file": "weights/best_qat.pt",
                "epoch": int(state["last_completed_epoch"]),
                "selection": "final_frozen_qat_checkpoint",
                "fitness": float(state["best_deployment_fitness"]),
            }
        metadata_path = Path(options.save_dir) / "qat_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def prepare_model(model) -> None:
        state["model"] = prepare_ac550_qat(model, arguments.qat_observer_momentum)
        write_metadata("TRAINING")

    def update_phase() -> None:
        epoch = int(state["epoch"])
        model = state["model"]
        phase = qat_phase_for_epoch(
            epoch,
            observer_epochs=arguments.qat_observer_epochs,
            activation_quant_epoch=arguments.qat_activation_epoch,
            freeze_observer_epoch=arguments.qat_freeze_observer_epoch,
        )
        set_qat_phase(model, phase)
        if should_freeze_batch_norm(epoch, freeze_bn_epoch=arguments.qat_freeze_bn_epoch):
            _freeze_batch_norm(model)
        if is_deployment_candidate_epoch(
            epoch,
            freeze_observer_epoch=arguments.qat_freeze_observer_epoch,
            freeze_bn_epoch=arguments.qat_freeze_bn_epoch,
        ):
            options.save_period = 1
        state["last_completed_epoch"] = epoch
        state["epoch"] = epoch + 1

    def record_deployment_fitness(log_vals, epoch, best_fitness, fitness) -> None:
        if not is_deployment_candidate_epoch(
            epoch,
            freeze_observer_epoch=arguments.qat_freeze_observer_epoch,
            freeze_bn_epoch=arguments.qat_freeze_bn_epoch,
        ):
            return
        score = scalar_fitness(fitness)
        if score > float(state["best_deployment_fitness"]):
            state["best_deployment_fitness"] = score
            state["best_deployment_epoch"] = int(epoch)

    callbacks.register_action("on_model_initialized", name="ac550-qat-prepare", callback=prepare_model)
    callbacks.register_action("on_train_epoch_start", name="ac550-qat-phase", callback=update_phase)
    callbacks.register_action("on_fit_epoch_end", name="ac550-qat-select", callback=record_deployment_fitness)
    vendor_train.main(options, callbacks=callbacks)

    save_dir = Path(options.save_dir)
    model = state.get("model")
    if model is None:
        raise RuntimeError("vendor training did not invoke the AC550 model preparation hook")
    candidate_start_epoch = max(arguments.qat_freeze_observer_epoch, arguments.qat_freeze_bn_epoch)
    if "best_deployment_epoch" not in state:
        raise RuntimeError("QAT completed without a frozen deployment candidate")
    selected_epoch = int(state["best_deployment_epoch"])
    candidate = copy_final_qat_checkpoint(
        save_dir,
        completed_epoch=int(state["last_completed_epoch"]),
        deployment_candidate_start_epoch=candidate_start_epoch,
        source_name=f"epoch{selected_epoch}.pt",
    )
    # Validation selects the EMA model stored in the epoch snapshot.  Mirror
    # YOLO's normal checkpoint finalization so deployment loads that exact EMA
    # QAT graph rather than the raw training model.
    strip_optimizer(candidate)
    state["deployment_checkpoint"] = "weights/best_qat.pt"
    state["last_completed_epoch"] = selected_epoch
    write_metadata("COMPLETED")
    return save_dir


def main() -> int:
    output = run(parse_args())
    print(f"AC550 QAT run: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
