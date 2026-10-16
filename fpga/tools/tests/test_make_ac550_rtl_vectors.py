"""Integration tests for the AC550 package-to-RTL vector compiler.

The checks intentionally execute the command-line tool against the exported
package.  They protect the deployable artifact boundary rather than its
implementation details: a bad mapper, incomplete executor accounting, or an
unchecked file range must make these tests fail.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPT = REPOSITORY / "fpga" / "tools" / "make_ac550_rtl_vectors.py"
EXPECTED_MANIFEST_SHA256 = "3BC1316D55C85B9010052E4ABCD1F886B74146DB7314CE2475EE6E45D016EE24"


def package_root() -> Path:
    configured = os.environ.get("AC550_PACKAGE_ROOT")
    assert configured, "set AC550_PACKAGE_ROOT to the exported AC550 package"
    root = Path(configured)
    assert (root / "manifest.json").is_file(), root
    return root


def run_compiler(package: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--package", str(package), "--output", str(output)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )


def test_compiler_emits_complete_runtime_boundary(tmp_path: Path) -> None:
    """Catches missing graph/vector mappings or a dropped supported executor."""
    result = run_compiler(package_root(), tmp_path / "generated")

    assert result.returncode == 0, result.stderr
    generated = tmp_path / "generated"
    lock = json.loads((generated / "package_lock.json").read_text(encoding="utf-8"))
    nodes = json.loads((generated / "runtime_nodes.json").read_text(encoding="utf-8"))

    assert lock["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert lock["graph_node_count"] == 87
    assert lock["convolution_count"] == 60
    assert lock["executor_counts"] == {
        "ARM_ADD": 7,
        "ARM_CONCAT": 13,
        "ARM_UPSAMPLE": 4,
        "FPGA_CONV_LINEAR": 3,
        "FPGA_CONV_RELU": 57,
        "FPGA_MAXPOOL": 3,
    }
    assert len(nodes) == 63
    assert (generated / "conv_layer_table.bin").stat().st_size == 63 * 64

    by_id = {node["graph_id"]: node for node in nodes}
    assert by_id["model.0"]["vector_source_module"] == "model.0.conv"
    assert by_id["model.0"]["kernel"] == [6, 6]
    assert by_id["model.0"]["input_shape"] == [1, 1, 256, 256]
    assert by_id["model.24.m.0"]["opcode"] == "CONV_LINEAR"
    assert by_id["model.24.m.0"]["vector_source_module"] == "model.24.m.0"
    assert by_id["model.9.maxpool.1"]["opcode"] == "MAXPOOL"
    assert by_id["model.9.maxpool.1"]["vector_source_module"] == "model.9.m"
    assert {tuple(node["kernel"]) for node in nodes} >= {(1, 1), (3, 3), (5, 5), (6, 6)}

    for node in nodes:
        assert (generated / node["input_mem"]).is_file()
        assert (generated / node["output_golden_mem"]).is_file()

    assert (generated / "model_0_weight.mem").is_file()
    assert (generated / "model_0_bias.mem").is_file()
    assert (generated / "model_0_multiplier.mem").is_file()
    assert (generated / "model_0_shift.mem").is_file()
    assert len((generated / "model_0_multiplier.mem").read_text().splitlines()) == 16
    assert len((generated / "model_0_shift.mem").read_text().splitlines()) == 16
    assert (generated / "requant_multiplier_i32.bin").stat().st_size == 4 * sum(
        node["output_shape"][1] for node in nodes if node["opcode"].startswith("CONV_")
    )
    assert (generated / "requant_shift_u8.bin").stat().st_size == sum(
        node["output_shape"][1] for node in nodes if node["opcode"].startswith("CONV_")
    )


def test_compiler_rejects_changed_lock_and_out_of_range_weight(tmp_path: Path) -> None:
    """Catches bypasses of the package identity and binary-range validation."""
    package = tmp_path / "package"
    shutil.copytree(package_root(), package)
    generated = tmp_path / "generated"

    first = run_compiler(package, generated)
    assert first.returncode == 0, first.stderr

    lock_path = generated / "package_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["manifest_sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    changed_lock = run_compiler(package, generated)
    assert changed_lock.returncode != 0
    assert "manifest SHA-256" in changed_lock.stderr

    shutil.rmtree(generated)
    params_path = package / "quant_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["convolutions"][0]["weight"]["offset"] = 1 << 31
    params_path.write_text(json.dumps(params), encoding="utf-8")
    bad_range = run_compiler(package, generated)
    assert bad_range.returncode != 0
    assert "weight range" in bad_range.stderr


def test_convolution_vectors_are_rescaled_to_package_activation_scales(tmp_path: Path) -> None:
    """Golden tensors are serialized with sample scales; RTL must see package scales."""
    package = package_root()
    generated = tmp_path / "generated"
    result = run_compiler(package, generated)
    assert result.returncode == 0, result.stderr

    quant = json.loads((package / "quant_params.json").read_text(encoding="utf-8"))
    golden = json.loads((package / "golden_vectors.json").read_text(encoding="utf-8"))
    node_id = "model.24.m.0"
    conv = next(item for item in quant["convolutions"] if item["source_module"] == node_id)
    vector = next(item for item in golden["vectors"] if item["source_module"] == node_id)
    source_scale = vector["input"]["quantization"]["scale"]
    target_scale = conv["input_quantization"]["scale"]
    source = (package / vector["input"]["file"]).read_bytes()
    expected = []
    for byte in source:
        signed = byte if byte < 128 else byte - 256
        expected.append(max(-127, min(127, round(signed * source_scale / target_scale))) & 0xFF)
    actual = [int(line, 16) for line in (generated / "model_24_m_0_input.mem").read_text().splitlines()]
    assert actual == expected
