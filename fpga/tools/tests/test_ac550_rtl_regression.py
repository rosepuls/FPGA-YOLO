from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_ac550_rtl_regression.py"
SPEC = importlib.util.spec_from_file_location("ac550_rtl_regression", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_signed_mem_parser_and_first_mismatch() -> None:
    values = MODULE.parse_i8_mem_text("00\n7F\n80\nFF\n")
    assert values == [0, 127, -128, -1]
    report = MODULE.compare_outputs([1, -2, 3], [1, -1, 4])
    assert report["count"] == 3
    assert report["mismatches"] == 2
    assert report["first_mismatch"] == {"index": 1, "actual": -2, "expected": -1}


def test_exact_output_report_contains_crc32() -> None:
    report = MODULE.compare_outputs([-128, 0, 127], [-128, 0, 127])
    assert report["mismatches"] == 0
    assert report["crc32_actual"] == report["crc32_expected"]


def test_integer_reference_uses_package_weights_bias_and_requant(tmp_path: Path) -> None:
    node = {
        "opcode": "CONV_RELU",
        "input_shape": [1, 2, 1, 1],
        "output_shape": [1, 1, 1, 1],
        "kernel": [1, 1], "stride": [1, 1], "padding": [0, 0],
        "input_mem": "input.mem", "weight_mem": "weight.mem", "bias_mem": "bias.mem",
        "multiplier_mem": "multiplier.mem", "shift_mem": "shift.mem",
    }
    (tmp_path / "input.mem").write_text("02\nFD\n")
    (tmp_path / "weight.mem").write_text("04\n05\n")
    (tmp_path / "bias.mem").write_text("0000000A\n")
    (tmp_path / "multiplier.mem").write_text("00000002\n")
    (tmp_path / "shift.mem").write_text("01\n")
    assert MODULE.integer_reference(node, tmp_path) == [3]
