# AC550 INT8 package RTL regression — v4

Date: 2026-10-16  
Status: PASS for the package-level RTL arithmetic regression; not a G2, Quartus, or board acceptance.

## Package identity

- Source checkpoint: `runs/yolov5n_gray_relu_256_qat_staged_v1/weights/best_qat.pt`
- Checkpoint SHA-256: `20DC478B4C097FEC313443E1936CD2339B6A100CC00F821E6DF9D9F30DFAD30B`
- Exported package: `runs/ac550_yolov5n_relu_int8_qat_staged_v4`
- Package manifest SHA-256: `30EC0F313A521A66492F86CB8E1C1265E6250C58326878F463936A57609093F4`
- v1 package was left unchanged. v4 contains the same quantized model parameters, with Golden tensors generated from the package's exact INT8 operator semantics.

## RTL regression

Configuration: `PIN=8`, `POUT=8`, `TILE_H=4`, `TILE_W=4`.

The package regression selected and completed 63 FPGA nodes: 57 Conv+ReLU, 3 linear Detect Conv, and 3 MaxPool. It compared 1,953,216 output bytes. Every node passed output-length and byte comparison checks. Both `rtl_vs_integer` and `package_golden_vs_integer` report zero mismatches for all outputs; CRC32 values agree for every node.

Command used:

```powershell
D:\Conda\conda_envs\zynq-defect\python.exe -u fpga/tools/run_ac550_rtl_regression.py --package runs/ac550_yolov5n_relu_int8_qat_staged_v4 --vectors fpga/sim/results/task4-tiled-8x8-v4/vectors --pin 8 --pout 8 --output fpga/sim/results/full-package-rtl-v4 --timeout-seconds 3600
```

Machine-readable run details are in the local ignored output `fpga/sim/results/full-package-rtl-v4/summary.json` (`selected=63`, `completed=63`, `passed=true`). The package and vector files are also local generated artifacts. On this checkout, verify the checkpoint, manifest, every package artifact, and the local 63-node summary with:

```powershell
pwsh -NoProfile -File fpga/tools/verify_ac550_v4_package.ps1
```

To rebuild ignored artifacts, use the same checkpoint and 64 sorted calibration images. The manifest records absolute image paths, so recreating the *identical manifest hash* requires the same checkout path; elsewhere, compare the per-artifact hashes and regenerate vectors against the newly exported manifest.

```powershell
$python = 'D:\Conda\conda_envs\zynq-defect\python.exe'
& $python models/export/export_yolov5n_relu_ac550.py --weights runs/yolov5n_gray_relu_256_qat_staged_v1/weights/best_qat.pt --calibration-dir data/LV-YOLO-ROI256/images/train --calibration-images 64 --output runs/ac550_yolov5n_relu_int8_qat_staged_v4_replay
& $python fpga/tools/make_ac550_rtl_vectors.py --package runs/ac550_yolov5n_relu_int8_qat_staged_v4_replay --output fpga/sim/results/v4-replay/vectors
& $python -u fpga/tools/run_ac550_rtl_regression.py --package runs/ac550_yolov5n_relu_int8_qat_staged_v4_replay --vectors fpga/sim/results/v4-replay/vectors --pin 8 --pout 8 --output fpga/sim/results/v4-replay/regression --timeout-seconds 3600
```

The exporter requires a new output directory. For the 14 Python tests and key ModelSim suites:

```powershell
$env:AC550_PACKAGE_ROOT = 'runs/ac550_yolov5n_relu_int8_qat_staged_v4'
& $python -m pytest models/tests/test_yolov5n_relu_export.py fpga/tools/tests -q
'cnn_job_controller_tb','fpga_op_executor_tb','tensor_dma_tb','conv2d_tiled_i8_tb' | ForEach-Object { pwsh -NoProfile -File fpga/sim/run_msim.ps1 -Testbench $_ }
```

## Additional regression

- Python exporter and package-tool tests: 14 passed, with `AC550_PACKAGE_ROOT=runs/ac550_yolov5n_relu_int8_qat_staged_v4`.
- ModelSim CSR, Requant/ReLU, tiled Conv parameter sweep, tensor DMA, Job Controller, and integrated executor tests passed.
- The integrated executor test covers Conv+ReLU → MaxPool → linear Conv in one DDR model, then a 64-byte output burst and an injected first-beat write error. It verifies contiguous burst-write addressing, exact task/frame IDs, and no later output beats after the error.
- `cnn_job_controller_tb` checks that a duplicate START while busy does not replace the active descriptor. `fpga_op_executor_tb` also connects the CSR completion/IRQ path to the executor; it checks IRQ stays low through output DMA completion, then rises with the matching task completion, and remains low for an aborted write.
- The executor testbench's DDR responder returns every word of a burst and advances addresses for each accepted write beat; this permits multiword DMA behavior to be checked rather than modeled as a single word.

## Scope and remaining gates

- This establishes bit-exact package operator behavior for the selected INT8 package. The package remains below the prior strict G2 IoU threshold; this report does not change that status or claim model accuracy acceptance.
- `fpga_op_executor` still stages complete tensors in local arrays around the tiled Conv core. Bounded tile-by-tile DDR streaming through the executor is not yet implemented or proven.
- Quartus synthesis, Cyclone V resource/timing closure, Platform Designer/HPS integration, IRQ wiring at the complete SoC boundary, and on-board execution have not been performed.
- Therefore the work is ready for the next RTL integration milestone, but is not yet a deployable FPGA bitstream.
