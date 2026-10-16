from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from run_ac550_ddr_regression import checked, read_output  # noqa: E402


def test_read_output_skips_address_markers_and_preserves_signed_bytes(tmp_path: Path) -> None:
    data = tmp_path / "out.mem"
    data.write_text("// DDR output\n@164000\n00\n7f\n80\nff\n", encoding="ascii")
    assert read_output(data) == [0, 127, -128, -1]


def test_modelsim_fatal_is_a_failure_even_when_process_exits_zero(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="simulation failed"):
        checked([sys.executable, "-c", "print('# ** Fatal: test failure')"], tmp_path)
