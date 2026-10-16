"""Run package-exported node vectors through the synthesizable RTL operators."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_i8_mem_text(text: str) -> list[int]:
    values = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        token = raw_line.strip()
        if not token or token.startswith("//"):
            continue
        if not re.fullmatch(r"[0-9A-Fa-f]{2}", token):
            raise ValueError(f"invalid INT8 mem token on line {line_number}: {token!r}")
        unsigned = int(token, 16)
        values.append(unsigned - 256 if unsigned >= 128 else unsigned)
    return values


def compare_outputs(actual: Sequence[int], expected: Sequence[int]) -> dict[str, object]:
    if len(actual) != len(expected):
        raise ValueError(f"output length differs: actual={len(actual)} expected={len(expected)}")
    mismatches = 0
    first_mismatch = None
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {"index": index, "actual": got, "expected": want}
    actual_crc = zlib.crc32(bytes(value & 0xFF for value in actual)) & 0xFFFFFFFF
    expected_crc = zlib.crc32(bytes(value & 0xFF for value in expected)) & 0xFFFFFFFF
    return {
        "count": len(actual),
        "mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "crc32_actual": f"{actual_crc:08X}",
        "crc32_expected": f"{expected_crc:08X}",
    }


def _read_i32_mem(path: Path) -> np.ndarray:
    values = [int(token, 16) for token in path.read_text(encoding="ascii").split()]
    return np.asarray(values, dtype=np.uint32).view(np.int32).astype(np.int64)


def integer_reference(node: dict, vector_dir: Path) -> list[int]:
    """Independent NumPy/host-integer implementation of the package's exact operator contract."""
    ic, ih, iw = node["input_shape"][1:]
    oc, oh, ow = node["output_shape"][1:]
    input_bytes = bytes(int(token, 16) for token in (vector_dir / node["input_mem"]).read_text().split())
    inputs = np.frombuffer(input_bytes, dtype=np.uint8).view(np.int8).astype(np.int64).reshape(ic, ih, iw)
    output = np.empty((oc, oh, ow), dtype=np.int64)
    if node["opcode"] == "MAXPOOL":
        output.fill(-127)
        for ky in range(5):
            for kx in range(5):
                for oy in range(oh):
                    iy = oy + ky - 2
                    if not 0 <= iy < ih:
                        continue
                    for ox in range(ow):
                        ix = ox + kx - 2
                        if 0 <= ix < iw:
                            output[:, oy, ox] = np.maximum(output[:, oy, ox], inputs[:, iy, ix])
        return output.reshape(-1).astype(np.int8).tolist()

    kernel, stride, padding = node["kernel"][0], node["stride"][0], node["padding"][0]
    weight_bytes = bytes(int(token, 16) for token in (vector_dir / node["weight_mem"]).read_text().split())
    weights = np.frombuffer(weight_bytes, dtype=np.uint8).view(np.int8).astype(np.int64).reshape(oc, ic, kernel, kernel)
    bias = _read_i32_mem(vector_dir / node["bias_mem"])
    multiplier = _read_i32_mem(vector_dir / node["multiplier_mem"])
    shifts = np.asarray([int(token, 16) for token in (vector_dir / node["shift_mem"]).read_text().split()], dtype=np.int64)
    accumulators = np.broadcast_to(bias[:, None, None], (oc, oh, ow)).copy()
    accumulators_flat = accumulators.reshape(oc, -1)
    output_y = np.arange(oh, dtype=np.int64)
    output_x = np.arange(ow, dtype=np.int64)
    for ky in range(kernel):
        for kx in range(kernel):
            input_y = output_y * stride + ky - padding
            input_x = output_x * stride + kx - padding
            valid_y = np.flatnonzero((input_y >= 0) & (input_y < ih))
            valid_x = np.flatnonzero((input_x >= 0) & (input_x < iw))
            if valid_y.size == 0 or valid_x.size == 0:
                continue
            patches = inputs[:, input_y[valid_y, None], input_x[None, valid_x]].reshape(ic, -1)
            output_positions = (valid_y[:, None] * ow + valid_x[None, :]).reshape(-1)
            accumulators_flat[:, output_positions] += weights[:, :, ky, kx] @ patches
    products = accumulators * multiplier[:, None, None]
    quantized = np.empty_like(accumulators)
    for channel in range(oc):
        shift = int(shifts[channel])
        product = products[channel]
        quotient = product >> shift
        if shift:
            remainder = product - (quotient << shift)
            half = 1 << (shift - 1)
            round_up = (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
            quotient = quotient + round_up.astype(np.int64)
        if node["opcode"] == "CONV_RELU":
            quotient = np.maximum(quotient, 0)
        quantized[channel] = np.clip(quotient, -127, 127)
    return quantized.reshape(-1).astype(np.int8).tolist()


def run_checked(command: list[str], *, cwd: Path, timeout: int | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-8000:]}\nstderr:\n{result.stderr[-8000:]}"
        )
    return result.stdout + result.stderr


def compile_simulator(repo: Path, build: Path, modelsim_bin: Path) -> None:
    build.mkdir(parents=True, exist_ok=True)
    if not (build / "_info").is_file():
        run_checked([str(modelsim_bin / "vlib.exe"), str(build)], cwd=repo)
    sources = [
        repo / "fpga/rtl/common/ac550_cnn_pkg.sv",
        repo / "fpga/rtl/cnn/conv2d_i8.sv",
        repo / "fpga/rtl/cnn/maxpool_i8.sv",
        repo / "fpga/sim/ac550_package_rtl_tb.sv",
    ]
    run_checked([str(modelsim_bin / "vlog.exe"), "-work", str(build), *map(str, sources)], cwd=repo)
    if not (build / "modelsim.ini").exists():
        run_checked([str(modelsim_bin / "vmap.exe"), "-c"], cwd=build)
    run_checked([str(modelsim_bin / "vmap.exe"), "work", str(build)], cwd=build)


def node_plusargs(node: dict, vector_dir: Path, actual_file: Path, max_cycles: int) -> list[str]:
    input_shape = node["input_shape"]
    output_shape = node["output_shape"]
    args = {
        "NODE": node["graph_id"],
        "OP": {"CONV_RELU": 1, "CONV_LINEAR": 2, "MAXPOOL": 3}[node["opcode"]],
        "IC": input_shape[1], "IH": input_shape[2], "IW": input_shape[3],
        "OC": output_shape[1], "OH": output_shape[2], "OW": output_shape[3],
        "KERNEL": node["kernel"][0], "STRIDE": node["stride"][0], "PADDING": node["padding"][0],
        "INPUT": str(vector_dir / node["input_mem"]),
        "GOLDEN": str(vector_dir / node["output_golden_mem"]),
        "ACTUAL": str(actual_file), "MAX_CYCLES": max_cycles,
    }
    if node["opcode"] != "MAXPOOL":
        args.update({
            "WEIGHT": str(vector_dir / node["weight_mem"]),
            "BIAS": str(vector_dir / node["bias_mem"]),
            "MULT": str(vector_dir / node["multiplier_mem"]),
            "SHIFT": str(vector_dir / node["shift_mem"]),
        })
    return [f"+{key}={value}" for key, value in args.items()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(os.environ.get("AC550_PACKAGE_ROOT", "")))
    parser.add_argument("--vectors", type=Path, default=Path("fpga/sim/generated"))
    parser.add_argument("--output", type=Path, default=Path("fpga/sim/results/package-rtl"))
    parser.add_argument("--modelsim-bin", type=Path, default=Path(r"D:\modeltech64_10.6e\win64"))
    parser.add_argument("--node", action="append", help="graph node ID; repeat to select several")
    parser.add_argument("--limit", type=int, help="run only the first N selected descriptors")
    parser.add_argument("--max-cycles", type=int, default=100_000_000)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    package = args.package.resolve()
    vectors = (repo / args.vectors).resolve() if not args.vectors.is_absolute() else args.vectors.resolve()
    output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    lock = json.loads((vectors / "package_lock.json").read_text(encoding="utf-8"))
    manifest_hash = __import__("hashlib").sha256((package / "manifest.json").read_bytes()).hexdigest().upper()
    if lock["manifest_sha256"] != manifest_hash:
        parser.error(f"vectors/package mismatch: {lock['manifest_sha256']} != {manifest_hash}")
    nodes = json.loads((vectors / "runtime_nodes.json").read_text(encoding="utf-8"))
    if args.node:
        requested = set(args.node)
        nodes = [node for node in nodes if node["graph_id"] in requested]
        missing = requested - {node["graph_id"] for node in nodes}
        if missing:
            parser.error(f"unknown graph node(s): {', '.join(sorted(missing))}")
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        nodes = nodes[:args.limit]
    if not nodes:
        parser.error("no descriptors selected")

    build = repo / "fpga/build/ac550-package-rtl"
    modelsim_bin = args.modelsim_bin.resolve()
    if not args.skip_compile:
        compile_simulator(repo, build, modelsim_bin)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, node in enumerate(nodes):
        actual = output / f"{node['descriptor_index']:02d}_{node['graph_id'].replace('.', '_')}_actual.mem"
        plusargs = node_plusargs(node, vectors, actual, args.max_cycles)
        command = [str(modelsim_bin / "vsim.exe"), "-c", "work.ac550_package_rtl_tb", *plusargs, "-do", "run -all; quit -f"]
        log = run_checked(command, cwd=build, timeout=args.timeout_seconds)
        (output / f"{node['descriptor_index']:02d}_{node['graph_id'].replace('.', '_')}.log").write_text(log, encoding="utf-8")
        actual_values = parse_i8_mem_text(actual.read_text(encoding="ascii"))
        expected_values = parse_i8_mem_text((vectors / node["output_golden_mem"]).read_text(encoding="ascii"))
        integer_values = integer_reference(node, vectors)
        rtl_report = compare_outputs(actual_values, integer_values)
        package_golden_report = compare_outputs(expected_values, integer_values)
        report = {
            "graph_id": node["graph_id"], "opcode": node["opcode"],
            "rtl_vs_integer": rtl_report,
            "package_golden_vs_integer": package_golden_report,
            "cycles": next((line.split("cycles=", 1)[1].split()[0] for line in log.splitlines() if "RTL_NODE_RESULT" in line), None),
        }
        results.append(report)
        print(json.dumps(report, ensure_ascii=False))
        if rtl_report["mismatches"]:
            break
        print(f"[{index + 1}/{len(nodes)}] {node['graph_id']}: RTL exactly matches independent integer reference")
    summary = {"manifest_sha256": manifest_hash, "selected": len(nodes), "completed": len(results), "passed": all(item["rtl_vs_integer"]["mismatches"] == 0 for item in results), "nodes": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"package RTL regression: {summary['completed']}/{summary['selected']} nodes, passed={summary['passed']}")
    return 0 if summary["passed"] and summary["completed"] == summary["selected"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
