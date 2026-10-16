"""Batch G2 acceptance for the AC550 package-backed INT8 simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Any

THIS_FILE = Path(__file__).resolve()
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from int8_inference_ac550 import _load_input, compare_context, load_comparison_context


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
HEAD_NAMES = ("P3", "P4", "P5")


def select_images(image_dir: Path, limit: int | None = None) -> list[Path]:
    """Return a stable, case-insensitive list of supported image files."""
    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise ValueError(f"image directory does not exist: {image_dir}")
    images = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("image limit must be positive")
        images = images[:limit]
    if not images:
        raise ValueError(f"no supported images found in: {image_dir}")
    return images


def _thresholds_from_args(arguments: argparse.Namespace) -> dict[str, float | int]:
    return {
        "max_head_mean_abs_diff": arguments.max_head_mean_abs_diff,
        "max_head_rmse": arguments.max_head_rmse,
        "min_box_iou": arguments.min_box_iou,
        "max_confidence_diff": arguments.max_confidence_diff,
        "min_images": arguments.min_images,
    }


def aggregate_reports(reports: list[dict[str, Any]], thresholds: dict[str, float | int]) -> dict[str, Any]:
    """Aggregate per-image evidence and evaluate explicit G2 acceptance rules."""
    failure_reasons: list[str] = []
    if len(reports) < int(thresholds["min_images"]):
        failure_reasons.append(f"only {len(reports)} images were evaluated; minimum is {thresholds['min_images']}")
    if not reports:
        return {
            "passed": False,
            "failure_reasons": failure_reasons or ["no image reports were produced"],
            "image_count": 0,
            "worst_heads": {},
            "worst_nodes": {},
            "detection_summary": {},
            "node_coverage": {},
        }

    manifest_shas = {str(report.get("package_manifest_sha256", "")) for report in reports}
    if len(manifest_shas) != 1:
        failure_reasons.append("reports do not use one consistent manifest SHA-256")
    statuses = {str(report.get("package_status", "")) for report in reports}
    if statuses != {"PRE_G2_STATIC_QUANTIZATION"}:
        failure_reasons.append(f"unsupported package statuses: {sorted(statuses)}")

    worst_heads: dict[str, dict[str, Any]] = {}
    for name in HEAD_NAMES:
        head_reports = [report["heads"][name] for report in reports]
        worst_mean = max(head_reports, key=lambda head: float(head["mean_abs_diff"]))
        worst_rmse = max(head_reports, key=lambda head: float(head["rmse"]))
        worst_heads[name] = {
            "max_mean_abs_diff": float(worst_mean["mean_abs_diff"]),
            "max_rmse": float(worst_rmse["rmse"]),
            "worst_mean_input": worst_mean.get("input", None),
            "worst_rmse_input": worst_rmse.get("input", None),
        }
        if worst_heads[name]["max_mean_abs_diff"] > float(thresholds["max_head_mean_abs_diff"]):
            failure_reasons.append(
                f"{name} mean_abs_diff {worst_heads[name]['max_mean_abs_diff']:.6f} exceeds "
                f"{thresholds['max_head_mean_abs_diff']}"
            )
        if worst_heads[name]["max_rmse"] > float(thresholds["max_head_rmse"]):
            failure_reasons.append(
                f"{name} rmse {worst_heads[name]['max_rmse']:.6f} exceeds {thresholds['max_head_rmse']}"
            )

    worst_nodes: dict[str, dict[str, Any]] = {}
    for report in reports:
        source = report["input"]["source"]
        for name, node in report.get("nodes", {}).items():
            candidate = {**node, "input": source}
            previous = worst_nodes.get(name)
            if previous is None or float(candidate["mean_abs_diff"]) > float(previous["mean_abs_diff"]):
                worst_nodes[name] = candidate

    matched = 0
    reference_count = 0
    quantized_count = 0
    unmatched_reference = 0
    unmatched_quantized = 0
    class_mismatches = 0
    ious: list[float] = []
    confidence_diffs: list[float] = []
    for report in reports:
        comparison = report["detections"]["comparison"]
        matched += int(comparison["matched"])
        reference_count += int(comparison["reference_count"])
        quantized_count += int(comparison["quantized_count"])
        unmatched_reference += len(comparison["unmatched_reference"])
        unmatched_quantized += len(comparison["unmatched_quantized"])
        class_mismatches += sum(not bool(match["class_match"]) for match in comparison["matches"])
        ious.extend(float(match["box_iou"]) for match in comparison["matches"])
        confidence_diffs.extend(float(match["confidence_abs_diff"]) for match in comparison["matches"])
    min_iou = min(ious) if ious else None
    max_confidence_diff = max(confidence_diffs) if confidence_diffs else 0.0
    if unmatched_reference or unmatched_quantized:
        failure_reasons.append(
            f"unmatched detections: reference={unmatched_reference}, quantized={unmatched_quantized}"
        )
    if class_mismatches:
        failure_reasons.append(f"class mismatches: {class_mismatches}")
    if min_iou is not None and min_iou < float(thresholds["min_box_iou"]):
        failure_reasons.append(f"minimum box IoU {min_iou:.6f} is below {thresholds['min_box_iou']}")
    if max_confidence_diff > float(thresholds["max_confidence_diff"]):
        failure_reasons.append(
            f"maximum confidence difference {max_confidence_diff:.6f} exceeds {thresholds['max_confidence_diff']}"
        )

    coverage = reports[0].get("graph", {}).get("node_coverage", {})
    if not bool(coverage.get("integer_simulation_complete", False)):
        failure_reasons.append("integer node simulation coverage is incomplete")
    if any(report.get("graph", {}).get("node_coverage") != coverage for report in reports):
        failure_reasons.append("node coverage differs between image reports")

    return {
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "image_count": len(reports),
        "manifest_sha256": next(iter(manifest_shas)) if len(manifest_shas) == 1 else None,
        "package_status": sorted(statuses),
        "worst_heads": worst_heads,
        "worst_nodes": worst_nodes,
        "detection_summary": {
            "reference_count": reference_count,
            "quantized_count": quantized_count,
            "matched": matched,
            "unmatched_reference": unmatched_reference,
            "unmatched_quantized": unmatched_quantized,
            "class_mismatches": class_mismatches,
            "min_box_iou": min_iou,
            "max_confidence_diff": max_confidence_diff,
        },
        "node_coverage": coverage,
    }


def run_acceptance(
    *,
    package: Path,
    weights: Path,
    relu_root: Path,
    images: list[Path],
    thresholds: dict[str, float | int],
    conf_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    context = load_comparison_context(package=package, weights=weights, relu_root=relu_root)
    reports: list[dict[str, Any]] = []
    for image in images:
        inputs, input_origin = _load_input(context.package, image)
        report = compare_context(
            context,
            inputs=inputs,
            input_origin=input_origin,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        for head in report["heads"].values():
            head["input"] = input_origin
        reports.append(report)
    aggregate = aggregate_reports(reports, thresholds)
    return {
        "schema_version": 1,
        "acceptance": "AC550_G2_INT8",
        "passed": aggregate["passed"],
        "failure_reasons": aggregate["failure_reasons"],
        "package": str(context.package),
        "manifest_sha256": _sha256_manifest(context.package),
        "package_status": context.manifest["status"],
        "thresholds": {
            **thresholds,
            "confidence": conf_threshold,
            "nms_iou": iou_threshold,
        },
        "images": [report["input"]["source"] for report in reports],
        "aggregate": aggregate,
        "per_image": reports,
    }


def _sha256_manifest(package: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with (package / "manifest.json").open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AC550 multi-image G2 INT8 acceptance.")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--relu-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--min-images", type=int, default=1)
    parser.add_argument("--max-head-mean-abs-diff", type=float, default=0.25)
    parser.add_argument("--max-head-rmse", type=float, default=0.35)
    parser.add_argument("--min-box-iou", type=float, default=0.90)
    parser.add_argument("--max-confidence-diff", type=float, default=0.05)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    images = select_images(arguments.image_dir, arguments.max_images)
    thresholds = _thresholds_from_args(arguments)
    report = run_acceptance(
        package=arguments.package.resolve(),
        weights=arguments.weights.resolve(),
        relu_root=arguments.relu_root.resolve(),
        images=images,
        thresholds=thresholds,
        conf_threshold=arguments.conf_thres,
        iou_threshold=arguments.iou_thres,
    )
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"AC550 G2 acceptance report: {output}")
    print(f"images: {report['aggregate']['image_count']}")
    print(f"passed: {report['passed']}")
    for reason in report["failure_reasons"]:
        print(f"failure: {reason}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
