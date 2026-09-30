"""Create YOLOv5 training curves and a test-set confusion matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = (
    "metrics/precision",
    "metrics/recall",
    "metrics/mAP_0.5",
    "metrics/mAP_0.5:0.95",
)
TRAIN_LOSS_COLUMNS = ("train/box_loss", "train/obj_loss", "train/cls_loss")
VALIDATION_LOSS_COLUMNS = ("val/box_loss", "val/obj_loss", "val/cls_loss")
REQUIRED_COLUMNS = ("epoch", *METRIC_COLUMNS, *TRAIN_LOSS_COLUMNS, *VALIDATION_LOSS_COLUMNS)


def _read_results(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"results CSV has no header: {csv_path}")
        rows = [
            {key.strip(): float(value.strip()) for key, value in row.items() if key and value is not None}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"results CSV has no epoch data: {csv_path}")
    missing = set(REQUIRED_COLUMNS) - set(rows[0])
    if missing:
        raise ValueError(f"results CSV is missing columns: {', '.join(sorted(missing))}")
    return rows


def _series(rows: Iterable[dict[str, float]], name: str) -> list[float]:
    return [row[name] for row in rows]


def create_training_plots(csv_path: Path, output_dir: Path) -> dict[str, Path]:
    """Render detection-metric and loss curves from a YOLOv5 results.csv file."""
    rows = _read_results(Path(csv_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = _series(rows, "epoch")

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    labels = {
        "metrics/precision": "Precision",
        "metrics/recall": "Recall",
        "metrics/mAP_0.5": "mAP@0.5",
        "metrics/mAP_0.5:0.95": "mAP@0.5:0.95",
    }
    for column in METRIC_COLUMNS:
        axis.plot(epochs, _series(rows, column), linewidth=2, label=labels[column])
    axis.set(title="YOLOv5 detection metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1.02))
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right")
    metrics_path = output_dir / "metrics.png"
    fig.savefig(metrics_path, dpi=200, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharex=True)
    for axis, columns, title in (
        (axes[0], TRAIN_LOSS_COLUMNS, "Training loss"),
        (axes[1], VALIDATION_LOSS_COLUMNS, "Validation loss"),
    ):
        for column in columns:
            axis.plot(epochs, _series(rows, column), linewidth=2, label=column.rsplit("/", 1)[-1])
        axis.set(title=title, xlabel="Epoch", ylabel="Loss")
        axis.grid(alpha=0.3)
        axis.legend()
    losses_path = output_dir / "losses.png"
    fig.savefig(losses_path, dpi=200, facecolor="white")
    plt.close(fig)

    return {"metrics": metrics_path, "losses": losses_path}


def build_validation_command(
    *,
    python: str,
    yolo_root: Path,
    weights: Path,
    data: Path,
    output_root: Path,
    device: str,
    batch_size: int,
    workers: int,
) -> list[str]:
    """Build the project's grayscale test-set validation command."""
    return [
        python,
        str(Path(yolo_root) / "val.py"),
        "--weights",
        str(weights),
        "--data",
        str(data),
        "--imgsz",
        "256",
        "--channels",
        "1",
        "--task",
        "test",
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--project",
        str(output_root),
        "--name",
        "validation",
        "--exist-ok",
    ]


def generate_confusion_matrix(
    *,
    python: str,
    yolo_root: Path,
    weights: Path,
    data: Path,
    output_root: Path,
    device: str,
    batch_size: int,
    workers: int,
) -> Path:
    """Run YOLOv5 test validation and return its generated confusion matrix path."""
    command = build_validation_command(
        python=python,
        yolo_root=yolo_root,
        weights=weights,
        data=data,
        output_root=output_root,
        device=device,
        batch_size=batch_size,
        workers=workers,
    )
    subprocess.run(command, cwd=yolo_root, check=True)
    matrix_path = output_root / "validation" / "confusion_matrix.png"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"YOLOv5 validation did not create: {matrix_path}")
    return matrix_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot YOLOv5 metrics/losses and generate a test-set confusion matrix."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="YOLOv5 training result directory")
    parser.add_argument("--data", type=Path, required=True, help="four-class YOLO dataset YAML")
    parser.add_argument("--weights", type=Path, help="defaults to <run-dir>/weights/best.pt")
    parser.add_argument("--output-dir", type=Path, help="defaults to <run-dir>/analysis")
    parser.add_argument(
        "--yolo-root", type=Path, default=PROJECT_ROOT / "third_party" / "yolov5", help="YOLOv5 source directory"
    )
    parser.add_argument("--device", default="0", help="CUDA device, or cpu")
    parser.add_argument("--batch-size", type=int, default=32, help="test validation batch size")
    parser.add_argument("--workers", type=int, default=4, help="test validation worker count")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    run_dir = arguments.run_dir.resolve()
    output_dir = (arguments.output_dir or run_dir / "analysis").resolve()
    weights = (arguments.weights or run_dir / "weights" / "best.pt").resolve()
    data = arguments.data.resolve()
    yolo_root = arguments.yolo_root.resolve()
    csv_path = run_dir / "results.csv"

    for path, description in ((csv_path, "results CSV"), (weights, "weights"), (data, "dataset YAML"), (yolo_root / "val.py", "YOLOv5 val.py")):
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")

    plots = create_training_plots(csv_path, output_dir)
    matrix = generate_confusion_matrix(
        python=sys.executable,
        yolo_root=yolo_root,
        weights=weights,
        data=data,
        output_root=output_dir,
        device=arguments.device,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
    )
    print(f"metrics: {plots['metrics']}")
    print(f"losses: {plots['losses']}")
    print(f"confusion_matrix: {matrix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
