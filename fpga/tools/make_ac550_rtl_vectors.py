"""Compile an exported AC550 INT8 package into deterministic RTL inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_EXECUTOR_COUNTS = {
    "ARM_ADD": 7,
    "ARM_CONCAT": 13,
    "ARM_UPSAMPLE": 4,
    "FPGA_CONV_LINEAR": 3,
    "FPGA_CONV_RELU": 57,
    "FPGA_MAXPOOL": 3,
}
OPCODES = {"FPGA_CONV_RELU": 1, "FPGA_CONV_LINEAR": 2, "FPGA_MAXPOOL": 3}
OPCODE_NAMES = {1: "CONV_RELU", 2: "CONV_LINEAR", 3: "MAXPOOL"}
RECORD = struct.Struct("<IBBBBHHHHHHIIIIIIHH16x")


class PackageError(ValueError):
    """Raised when the exported package cannot be executed safely by RTL."""


@dataclass(frozen=True)
class Vector:
    source_module: str
    input_file: str
    output_file: str
    input_shape: list[int]
    output_shape: list[int]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PackageError(f"missing package file: {path.name}") from error
    except json.JSONDecodeError as error:
        raise PackageError(f"invalid JSON in {path.name}: {error}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def write_mem(path: Path, payload: bytes) -> None:
    path.write_text("".join(f"{byte:02X}\n" for byte in payload), encoding="ascii")


def checked_slice(package: Path, description: str, file_name: str, offset: int, byte_count: int) -> bytes:
    file_path = package / file_name
    try:
        size = file_path.stat().st_size
    except FileNotFoundError as error:
        raise PackageError(f"missing {description} file: {file_name}") from error
    if offset < 0 or byte_count < 0 or offset + byte_count > size:
        raise PackageError(
            f"{description} range [{offset}, {offset + byte_count}) exceeds {file_name} size {size}"
        )
    with file_path.open("rb") as source:
        source.seek(offset)
        return source.read(byte_count)


def normalize_convolution_name(graph_node: dict[str, Any], names: set[str]) -> str:
    source = graph_node["source_module"]
    if source in names:
        return source
    candidate = f"{source}.conv"
    if candidate in names:
        return candidate
    raise PackageError(f"no quantized convolution matches graph node {graph_node['id']}")


def maxpool_vector_name(graph_node: dict[str, Any]) -> str:
    try:
        call_index = int(graph_node["id"].rsplit(".", 1)[1])
    except (IndexError, ValueError) as error:
        raise PackageError(f"invalid MaxPool graph node id: {graph_node['id']}") from error
    source = graph_node["source_module"]
    return source if call_index == 1 else f"{source}#{call_index}"


def vector_map(golden: dict[str, Any]) -> dict[str, Vector]:
    vectors: dict[str, Vector] = {}
    for entry in golden["vectors"]:
        source = entry["source_module"]
        if source in vectors:
            raise PackageError(f"duplicate golden vector for {source}")
        vectors[source] = Vector(
            source_module=source,
            input_file=entry["input"]["file"],
            output_file=entry["output"]["file"],
            input_shape=entry["input"]["shape"],
            output_shape=entry["output"]["shape"],
        )
    return vectors


def validate_existing_lock(output: Path, manifest_hash: str) -> None:
    lock_path = output / "package_lock.json"
    if not lock_path.is_file():
        return
    recorded = load_json(lock_path).get("manifest_sha256")
    if recorded != manifest_hash:
        raise PackageError(
            f"manifest SHA-256 differs from existing package_lock.json: {recorded} != {manifest_hash}"
        )


def compile_package(package: Path, output: Path) -> list[dict[str, Any]]:
    manifest_path = package / "manifest.json"
    manifest_hash = sha256(manifest_path)
    validate_existing_lock(output, manifest_hash)
    manifest = load_json(manifest_path)
    graph = load_json(package / "graph.json")
    quant = load_json(package / "quant_params.json")
    golden = load_json(package / "golden_vectors.json")
    graph_nodes = graph["nodes"]
    executor_counts = {name: sum(node["executor"] == name for node in graph_nodes) for name in EXPECTED_EXECUTOR_COUNTS}
    if len(graph_nodes) != 87 or executor_counts != EXPECTED_EXECUTOR_COUNTS:
        raise PackageError(f"unexpected graph partition: {len(graph_nodes)} nodes, {executor_counts}")
    convolutions = quant["convolutions"]
    if len(convolutions) != 60:
        raise PackageError(f"expected 60 quantized convolutions, found {len(convolutions)}")
    conv_by_source = {item["source_module"]: item for item in convolutions}
    if len(conv_by_source) != len(convolutions):
        raise PackageError("duplicate quantized convolution source module")
    vectors = vector_map(golden)

    output.mkdir(parents=True, exist_ok=True)
    multiplier_bytes = bytearray()
    shift_bytes = bytearray()
    table = bytearray()
    runtime_nodes: list[dict[str, Any]] = []
    descriptor_index = 0
    for graph_node in graph_nodes:
        executor = graph_node["executor"]
        if executor not in OPCODES:
            continue
        opcode = OPCODES[executor]
        if executor == "FPGA_MAXPOOL":
            source = maxpool_vector_name(graph_node)
            weight_offset = weight_bytes = bias_offset = bias_bytes = 0
            multiplier_offset = len(multiplier_bytes)
            shift_offset = len(shift_bytes)
        else:
            source = normalize_convolution_name(graph_node, set(conv_by_source))
            layer = conv_by_source[source]
            weight = layer["weight"]
            bias = layer["bias"]
            weight_offset, weight_bytes = weight["offset"], weight["bytes"]
            bias_offset, bias_bytes = bias["offset"], bias["bytes"]
            write_mem(output / f"{safe_stem(graph_node['id'])}_weight.mem", checked_slice(package, "weight", weight["file"], weight_offset, weight_bytes))
            write_mem(output / f"{safe_stem(graph_node['id'])}_bias.mem", checked_slice(package, "bias", bias["file"], bias_offset, bias_bytes))
            requant = layer["requantization"]
            multipliers = requant["integer_multiplier"]
            shifts = requant["right_shift"]
            if len(multipliers) != graph_node["output_channels"] or len(shifts) != len(multipliers):
                raise PackageError(f"requant table channel count mismatch for {source}")
            multiplier_offset = len(multiplier_bytes)
            shift_offset = len(shift_bytes)
            multiplier_bytes.extend(struct.pack(f"<{len(multipliers)}i", *multipliers))
            shift_bytes.extend(shifts)
        if source not in vectors:
            raise PackageError(f"missing input/output golden vector for {source}")
        vector = vectors[source]
        input_file = package / vector.input_file
        output_file = package / vector.output_file
        input_data = checked_slice(package, "golden input", vector.input_file, 0, input_file.stat().st_size)
        output_data = checked_slice(package, "golden output", vector.output_file, 0, output_file.stat().st_size)
        stem = safe_stem(graph_node["id"])
        input_mem = f"{stem}_input.mem"
        output_mem = f"{stem}_output_golden.mem"
        write_mem(output / input_mem, input_data)
        write_mem(output / output_mem, output_data)
        kernel, stride, padding = graph_node["kernel"], graph_node["stride"], graph_node["padding"]
        if kernel[0] != kernel[1] or stride[0] != stride[1] or padding[0] != padding[1]:
            raise PackageError(f"asymmetric spatial parameter for {graph_node['id']}")
        cin, hin, win = vector.input_shape[1:]
        cout, hout, wout = vector.output_shape[1:]
        table.extend(RECORD.pack(descriptor_index, opcode, kernel[0], stride[0], padding[0], cin, hin, win, cout, hout, wout, weight_offset, weight_bytes, bias_offset, bias_bytes, multiplier_offset, shift_offset, cout if opcode != OPCODES["FPGA_MAXPOOL"] else 0, 0))
        runtime_nodes.append({
            "descriptor_index": descriptor_index,
            "graph_id": graph_node["id"],
            "opcode": OPCODE_NAMES[opcode],
            "vector_source_module": source,
            "input_shape": vector.input_shape,
            "output_shape": vector.output_shape,
            "kernel": kernel,
            "stride": stride,
            "padding": padding,
            "weight_offset": weight_offset,
            "weight_bytes": weight_bytes,
            "bias_offset": bias_offset,
            "bias_bytes": bias_bytes,
            "multiplier_offset": multiplier_offset,
            "shift_offset": shift_offset,
            "input_mem": input_mem,
            "output_golden_mem": output_mem,
        })
        descriptor_index += 1
    if descriptor_index != 63:
        raise PackageError(f"expected 63 FPGA descriptors, generated {descriptor_index}")
    (output / "conv_layer_table.bin").write_bytes(table)
    (output / "requant_multiplier_i32.bin").write_bytes(multiplier_bytes)
    (output / "requant_shift_u8.bin").write_bytes(shift_bytes)
    (output / "runtime_nodes.json").write_text(json.dumps(runtime_nodes, indent=2) + "\n", encoding="utf-8")
    lock = {
        "manifest_sha256": manifest_hash,
        "package_status": manifest.get("status"),
        "graph_node_count": len(graph_nodes),
        "convolution_count": len(convolutions),
        "executor_counts": executor_counts,
        "descriptor_count": descriptor_index,
        "record_bytes": RECORD.size,
    }
    (output / "package_lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return runtime_nodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        nodes = compile_package(args.package.resolve(), args.output.resolve())
    except PackageError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"generated {len(nodes)} FPGA descriptors in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
