from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Sequence

try:
    from .gc10_coco.coco import ConversionConfig, build_dataset, validate_coco
    from .gc10_coco.source import parse_source
    from .gc10_coco.split import split_records, summarize_splits
except ImportError:
    from gc10_coco.coco import ConversionConfig, build_dataset, validate_coco
    from gc10_coco.source import parse_source
    from gc10_coco.split import split_records, summarize_splits


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / "Data" / "GC10-DET"
DEFAULT_OUTPUT = SCRIPT_DIR / "Data" / "GC10-DET-COCO"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit GC10-DET or convert it into a cleaned, augmented COCO dataset."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--convert", action="store_true", help="Write the COCO dataset; otherwise audit only.")
    parser.add_argument("--validate", type=Path, help="Validate an existing converted dataset and exit.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument("--negative-crop-size", type=int, default=512)
    parser.add_argument("--negative-margin", type=int, default=24)
    parser.add_argument("--negative-attempts", type=int, default=100)
    parser.add_argument("--negative-min-std", type=float, default=3.0)
    parser.add_argument("--minority-target", type=int)
    parser.add_argument("--max-augmentations-per-image", type=int, default=3)
    parser.add_argument("--no-negative-augmentation", action="store_true")
    parser.add_argument("--no-minority-augmentation", action="store_true")
    return parser


def _config(arguments: argparse.Namespace) -> ConversionConfig:
    return ConversionConfig(
        source_dir=arguments.source,
        output_dir=arguments.output,
        seed=arguments.seed,
        ratios=(arguments.train_ratio, arguments.val_ratio, arguments.test_ratio),
        overwrite=arguments.overwrite,
        negative_ratio=arguments.negative_ratio,
        negative_crop_size=arguments.negative_crop_size,
        negative_margin=arguments.negative_margin,
        negative_attempts=arguments.negative_attempts,
        negative_min_std=arguments.negative_min_std,
        minority_target=arguments.minority_target,
        max_augmentations_per_image=arguments.max_augmentations_per_image,
        enable_negative_augmentation=not arguments.no_negative_augmentation,
        enable_minority_augmentation=not arguments.no_minority_augmentation,
    )


def audit_source(config: ConversionConfig) -> dict:
    records, quarantine, unused = parse_source(config.source_dir)
    splits = split_records(records, config.ratios, config.seed)
    class_instances: Counter[str] = Counter()
    for record in records:
        class_instances.update(box.category_name for box in record.boxes)
    quarantine_reasons = Counter(item.reason for item in quarantine)
    return {
        "source": str(config.source_dir.resolve()),
        "source_xml": len(records) + len(quarantine),
        "accepted_source_images": len(records),
        "quarantined_records": len(quarantine),
        "quarantine_reasons": dict(sorted(quarantine_reasons.items())),
        "unused_physical_images": len(unused),
        "class_instances": dict(sorted(class_instances.items())),
        "split_summary": summarize_splits(splits),
        "output_created": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.validate is not None:
            report = validate_coco(arguments.validate)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["valid"] else 1
        config = _config(arguments)
        if arguments.convert:
            report = build_dataset(config)
        else:
            report = audit_source(config)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"error": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
