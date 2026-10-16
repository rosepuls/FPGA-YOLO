# AC550 v4 package — tile-streamed CSR/DDR regression

Date: 2026-10-16
Status: PASS in ModelSim behavioral simulation; not a Quartus timing/resource or board acceptance.

## Identity and reproducibility

- Source package: `runs/ac550_yolov5n_relu_int8_qat_staged_v4`
- Manifest SHA-256: `30EC0F313A521A66492F86CB8E1C1265E6250C58326878F463936A57609093F4`
- Checkpoint SHA-256: `20DC478B4C097FEC313443E1936CD2339B6A100CC00F821E6DF9D9F30DFAD30B`
- Verify the package and source checkpoint with `pwsh -NoProfile -File fpga/tools/verify_ac550_v4_package.ps1`. Regeneration instructions for ignored package/vector artifacts are in [the arithmetic baseline](ac550-int8-package-regression-v4.md).

Run the complete CSR-start/DDR transfer/8×8-lane tile-executor regression:

```powershell
D:\Conda\conda_envs\zynq-defect\python.exe -u fpga/tools/run_ac550_ddr_regression.py --package runs/ac550_yolov5n_relu_int8_qat_staged_v4 --vectors fpga/sim/results/task4-tiled-8x8-v4/vectors --output fpga/sim/results/ddr-wrapper-v4-final --timeout-seconds 3600 --max-cycles 100000000
```

For the three first-gate nodes, add `--node model.0 --node model.9.maxpool.1 --node model.24.m.2`. The command compiles the synthesizable `ac550_cnn_accelerator` CSR-to-executor wrapper and RTL testbench, writes each node's tensor/parameters into modeled DDR, programs the CSR descriptor and START, waits for matching task/frame completion and IRQ, then compares the output DDR image against both the package Golden and independent integer reference. A simulator `$fatal` is treated as failure even if ModelSim returns process code 0.

## Result

- All 63/63 FPGA nodes passed: 57 Conv+ReLU, 3 MaxPool, 3 Detect linear Conv.
- 1,953,216 output bytes compared; zero mismatches versus both oracles.
- Total modeled execution: 136,084,729 cycles, 16,252,260 read commands, 488,304 write commands across the independently launched nodes. These are simulation counters, not hardware performance measurements.
- First-gate nodes: `model.0` 6×6 Conv 262,144 bytes; `model.9.maxpool.1` 5×5 MaxPool 8,192 bytes; `model.24.m.2` linear Conv 1,728 bytes; each byte-exact.
- Machine-readable per-node results are in ignored `fpga/sim/results/ddr-wrapper-v4-final/summary.json`.

## Implementation and additional checks

The executor now has fixed-size banks: up to 1,152 feature bytes, 2,304 weight bytes, 512 accumulator bytes, 72 bias/multiplier/shift bytes, and 128 output bytes at `PIN=POUT=8`, `TILE_H=TILE_W=4`. Feature rows and OIHW weight slices are fetched by DMA per tile; output rows are written back immediately. No local array is sized by full input/weight/output tensor capacities. Descriptor bases stay 64-byte aligned, while tile row transfers can begin at any byte offset; the DMA test checks unaligned heads, full bursts, tails, and unchanged adjacent bytes.

The small integrated test still passes Conv→MaxPool→linear Conv, output write error abort, watchdog drain/abort and restart (including a one-cycle limit), address-range/overlap rejection, task/frame IDs, and IRQ ordering. The CSR test rejects high-bit dimensions that would truncate at the 16-bit executor interface; the DMA test covers one-byte unaligned writes at lanes 1–3 without adjacent-byte damage and rejects new transfers after either a read-command error or an accepted-read timeout until reset. The controller, DMA, tiled core parameter sweep, and 16 Python package/tool tests pass.

## Remaining gates

This does not establish Cyclone V fit, RAM inference, Fmax, HPS memory-system throughput, physical IRQ integration, board execution, or the strict G2 detection-accuracy gate. The current straightforward scheduler reloads some feature tiles for each output-channel block; optimize memory reuse only after measuring area and bandwidth. The behavioral DDR model also does not emulate a real HPS cache-coherency path.
