"""Run exported FPGA nodes through CSR, tile executor, DMA, and DDR model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from run_ac550_rtl_regression import compare_outputs, integer_reference, node_plusargs


def checked(command: list[str], cwd: Path, timeout: int | None = None) -> str:
    completed = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    output = completed.stdout + completed.stderr
    if completed.returncode or re.search(r"(?m)^# \*\* (?:Error|Fatal):", output):
        raise RuntimeError(f"simulation failed: {' '.join(command)}\n{output[-6000:]}")
    return output


def compile_simulator(repo: Path, build: Path, modelsim_bin: Path) -> None:
    build.mkdir(parents=True, exist_ok=True)
    if not (build / "_info").exists():
        checked([str(modelsim_bin / "vlib.exe"), str(build)], repo)
    sources = [
        "fpga/rtl/common/ac550_cnn_pkg.sv",
        "fpga/rtl/avalon/cnn_job_controller.sv",
        "fpga/rtl/avalon/ac550_cnn_csr.sv",
        "fpga/rtl/cnn/tensor_dma.sv",
        "fpga/rtl/cnn/mac_array_i8.sv",
        "fpga/rtl/cnn/requant_i32_to_i8.sv",
        "fpga/rtl/cnn/fpga_op_executor.sv",
        "fpga/rtl/avalon/ac550_cnn_accelerator.sv",
        "fpga/sim/avalon_ddr_model.sv",
        "fpga/sim/ac550_package_ddr_tb.sv",
    ]
    checked([str(modelsim_bin / "vlog.exe"), "-work", str(build), *(str(repo / source) for source in sources)], repo)
    if not (build / "modelsim.ini").exists():
        checked([str(modelsim_bin / "vmap.exe"), "-c"], build)
    checked([str(modelsim_bin / "vmap.exe"), "work", str(build)], build)


def read_output(path: Path) -> list[int]:
    values = []
    for line in path.read_text(encoding="ascii").splitlines():
        token = line.strip()
        if not token or token.startswith(("//", "@")):
            continue
        if not re.fullmatch(r"[0-9a-fA-F]{2}", token):
            raise ValueError(f"bad DDR output token: {token!r}")
        value = int(token, 16)
        values.append(value - 256 if value >= 128 else value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node", action="append", help="graph ID; omit to run all 63")
    parser.add_argument("--modelsim-bin", type=Path, default=Path("D:/modeltech64_10.6e/win64"))
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-cycles", type=int, default=100_000_000)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    package, vectors, output = args.package.resolve(), args.vectors.resolve(), args.output.resolve()
    manifest_hash = hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest().upper()
    lock = json.loads((vectors / "package_lock.json").read_text(encoding="utf-8"))
    if lock["manifest_sha256"] != manifest_hash:
        parser.error("package and vectors have different manifest hashes")
    all_nodes = json.loads((vectors / "runtime_nodes.json").read_text(encoding="utf-8"))
    wanted = set(args.node or [])
    nodes = [node for node in all_nodes if not wanted or node["graph_id"] in wanted]
    if len(nodes) != (len(wanted) if wanted else 63):
        parser.error("requested node missing or vector set incomplete")
    output.mkdir(parents=True, exist_ok=True)
    build = output / "modelsim"
    compile_simulator(repo, build, args.modelsim_bin.resolve())
    results = []
    for index, node in enumerate(nodes, 1):
        actual_file = output / f"{node['graph_id'].replace('.', '_')}.mem"
        plusargs = node_plusargs(node, vectors, actual_file, args.max_cycles)
        command = [
            str(args.modelsim_bin.resolve() / "vsim.exe"), "work.ac550_package_ddr_tb",
            "-c", *plusargs, "-do", "run -all; quit -f",
        ]
        try:
            simulation = checked(command, build, timeout=args.timeout_seconds)
        except Exception as exc:
            (output / "last_failure.log").write_text(str(exc), encoding="utf-8")
            raise
        match = re.search(
            r"DDR_NODE_RESULT node=(\S+) cycles=(\d+) outputs=(\d+) mismatches=(\d+) first=(-?\d+) reads=(\d+) writes=(\d+)",
            simulation,
        )
        if match is None or int(match.group(4)) != 0:
            raise RuntimeError(f"no passing DDR result for {node['graph_id']}\n{simulation[-4000:]}")
        actual = read_output(actual_file)
        expected = integer_reference(node, vectors)
        comparison = compare_outputs(actual, expected)
        golden = read_output(vectors / node["output_golden_mem"])
        golden_comparison = compare_outputs(actual, golden)
        if comparison["mismatches"] or golden_comparison["mismatches"]:
            raise RuntimeError(f"DDR mismatch for {node['graph_id']}: {comparison}, {golden_comparison}")
        results.append({
            "graph_id": node["graph_id"],
            "opcode": node["opcode"],
            "cycles": int(match.group(2)),
            "outputs": len(actual),
            "read_commands": int(match.group(6)),
            "write_commands": int(match.group(7)),
            "rtl_vs_integer": comparison,
            "rtl_vs_package_golden": golden_comparison,
        })
        print(f"[{index}/{len(nodes)}] {node['graph_id']}: {len(actual)} DDR bytes exact, {match.group(2)} cycles", flush=True)
    summary = {
        "manifest_sha256": manifest_hash,
        "selected": len(nodes),
        "completed": len(results),
        "passed": True,
        "bytes_compared": sum(item["outputs"] for item in results),
        "nodes": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"DDR regression passed: {len(results)}/{len(nodes)} nodes, {summary['bytes_compared']} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
