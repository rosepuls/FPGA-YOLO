# AC550 INT8 CNN RTL Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement an AC550 HPS-descriptor-driven INT8 accelerator for the 60 convolution nodes and 3 MaxPool nodes in the staged QAT package, with bit-exact comparison against the exported Golden tensors.

**Architecture:** ARM/HPS owns graph scheduling, DDR buffer lifetime, Add/Concat/Upsample, Decode and NMS. FPGA owns a reusable tiled operator engine selected by an opcode: Conv+ReLU, linear Detect Conv, or SPPF MaxPool. HPS submits one descriptor per FPGA node; FPGA performs DMA, executes the node, writes the tensor back to DDR and reports task/frame completion.

**Tech Stack:** SystemVerilog, Avalon-MM, ModelSim 10.6e, Quartus/Platform Designer, Python 3 vector-generation and regression tools.

## Package Baseline

The implementation input is runs/ac550_yolov5n_relu_int8_qat_staged_v1. Its manifest status is PRE_G2_STATIC_QUANTIZATION; use it for engineering bring-up and Golden comparison, not as a final accuracy release.

The package contains 87 graph nodes: 57 FPGA_CONV_RELU, 3 FPGA_CONV_LINEAR, 3 FPGA_MAXPOOL, 7 ARM_ADD, 13 ARM_CONCAT and 4 ARM_UPSAMPLE. The graph covers 1x1, 3x3 and 6x6 kernels, stride 1/2, padding 0/1/2, groups=1, maximum Cin=512 and maximum Cout=256.

The package manifest SHA-256 is 3BC1316D55C85B9010052E4ABCD1F886B74146DB7314CE2475EE6E45D016EE24. Weight storage is INT8 OIHW with per-output-channel scales; bias storage is INT32; activation storage is signed symmetric INT8 with zero point 0; quantization uses ROUND_HALF_TO_EVEN and saturation [-127,127].

## Global Constraints

- Target device: Cyclone V SoC 5CSXFC6C6U23I7.
- Initial CNN clock target: 100 MHz.
- Tensor layout: NCHW, contiguous row-major.
- Weight layout: OIHW, output-channel axis first.
- Normal Conv path: INT8 MAC, INT32 bias/accumulator, requant, then ReLU.
- Detect Conv path: INT8 MAC, INT32 bias/accumulator, requant, no ReLU.
- MaxPool path: signed INT8 comparison; padding value is -127; MaxPool does not imply ReLU.
- Feature, weight and output DDR addresses are 64-byte aligned.
- Bias and quantization-table addresses are at least 4-byte aligned; do not assume bias offsets are 64-byte aligned.
- Requant uses a signed 64-bit product and ROUND_HALF_TO_EVEN.
- FPGA must reject unsupported kernel, stride, padding, shape, address or byte-count combinations before writing output.
- RTL regression is bit-exact; no numerical tolerance is accepted for tensor bytes.
- Do not change the package quantization formula or tensor layout to improve timing.

---

## Task 1: Compile the model package into RTL runtime tables

**Files:**
- Create: fpga/tools/make_ac550_rtl_vectors.py
- Create: fpga/tools/tests/test_make_ac550_rtl_vectors.py
- Create: fpga/contracts/ac550_int8_rtl_contract.md
- Create: fpga/sim/generated/.gitignore

**Purpose:** The RTL and HPS runtime do not parse graph.json or quant_params.json. Convert the package into deterministic descriptor records, requant tables and ModelSim memory images.

**Produced artifacts:**

    fpga/sim/generated/package_lock.json
    fpga/sim/generated/conv_layer_table.bin
    fpga/sim/generated/requant_multiplier_i32.bin
    fpga/sim/generated/requant_shift_u8.bin
    fpga/sim/generated/model_0_input.mem
    fpga/sim/generated/model_0_weight.mem
    fpga/sim/generated/model_0_bias.mem
    fpga/sim/generated/model_0_output_golden.mem
    fpga/sim/generated/<node>_input.mem
    fpga/sim/generated/<node>_output_golden.mem

Each runtime table record contains node id, opcode, Cin/Hin/Win, Cout/Hout/Wout, kernel, stride, padding, weight offset/bytes, bias offset/bytes, multiplier offset and shift offset.

The mapper must normalize graph names such as model.0 to quant/vector names such as model.0.conv. Detect nodes model.24.m.0, model.24.m.1 and model.24.m.2 must remain linear and must not receive a .conv suffix.

**Tests:**
- Parse exactly 87 graph nodes and 60 convolution entries.
- Match every convolution node to an input and output Golden vector.
- Verify the executor counts 57/3/3/7/13/4.
- Verify weight and bias offsets plus byte counts stay within their binary files.
- Reject a package whose manifest SHA differs from package_lock.json.
- Generate representative cases for 6x6, 3x3, 1x1, Detect Linear and MaxPool.

- [ ] Write the failing package-schema and mapping tests.
- [ ] Run the focused Python tests and verify failure before implementation.
- [ ] Implement package parsing, node-name normalization and deterministic runtime-table generation.
- [ ] Run the focused tests and inspect generated offsets, shapes and hashes.

## Task 2: Define common RTL types, opcodes, errors and CSR

**Files:**
- Create: fpga/rtl/common/ac550_cnn_pkg.sv
- Create: fpga/rtl/avalon/ac550_cnn_csr.sv
- Create: fpga/sim/ac550_cnn_csr_tb.sv
- Create: fpga/sim/run_msim.ps1

Use these opcodes:

    OP_CONV_RELU   = 3'd1
    OP_CONV_LINEAR = 3'd2
    OP_MAXPOOL     = 3'd3

Use these error codes:

    0  OK
    1  BUSY_START
    2  ADDRESS_ALIGNMENT
    3  INVALID_SHAPE
    4  UNSUPPORTED_OPCODE
    5  UNSUPPORTED_KERNEL
    6  DMA_READ_ERROR
    7  DMA_WRITE_ERROR
    8  BYTE_COUNT_MISMATCH
    9  WATCHDOG_TIMEOUT
    10 TASK_ID_MISMATCH

CSR map:

    0x00 VERSION
    0x04 CONTROL: START, IRQ_EN, CLEAR
    0x08 STATUS: BUSY, DONE, ERROR
    0x0C ERROR_CODE
    0x10 OPCODE
    0x14 TASK_ID
    0x18 FRAME_ID
    0x1C DONE_TASK_ID
    0x20 INPUT_ADDR
    0x24 WEIGHT_ADDR
    0x28 BIAS_ADDR
    0x2C MULTIPLIER_ADDR
    0x30 SHIFT_ADDR
    0x34 OUTPUT_ADDR
    0x38 INPUT_BYTES
    0x3C OUTPUT_BYTES
    0x40-0x54 INPUT/OUTPUT C,H,W
    0x58 KERNEL
    0x5C STRIDE
    0x60 PADDING
    0x64 QUANT_ID
    0x68 WATCHDOG_CYCLES
    0x6C PERF_CYCLES

**Tests:**
- Atomically latch all descriptor fields on START.
- Reject START while BUSY.
- Clear DONE and ERROR without changing the latched configuration.
- Reject illegal opcode, shape and address alignment before DMA.
- Raise IRQ only after output write completion and expose the matching DONE_TASK_ID.

- [ ] Write the CSR testbench for normal start/done, busy start, clear and error paths.
- [ ] Run the ModelSim CSR test.
- [ ] Implement the package typedefs, CSR registers and sticky status bits.
- [ ] Re-run the test and inspect waveforms for descriptor latching.

## Task 3: Implement bit-exact Requant and ReLU arithmetic

**Files:**
- Create: fpga/rtl/cnn/requant_i32_to_i8.sv
- Create: fpga/rtl/cnn/relu_i8.sv
- Create: fpga/sim/requant_i32_to_i8_tb.sv
- Create: fpga/sim/relu_i8_tb.sv

For each output channel, compute:

    product = signed_int64(acc_i32 * multiplier_i32)
    q = round_half_to_even(product / 2^right_shift)
    q = clamp(q, -127, 127)
    CONV_RELU: output = max(q, 0)
    CONV_LINEAR: output = q

The implementation must compare remainder against half and increment only when remainder is greater than half or when remainder equals half and the quotient is odd.

Tests cover positive and negative products, exact ties, even/odd quotient ties, right shifts 35 and 43, saturation, ReLU negative values and Detect Linear negative values.

- [ ] Write failing arithmetic tests against a Python reference generated from the package rounding rule.
- [ ] Run the focused ModelSim tests and verify failure.
- [ ] Implement signed 64-bit requant, saturation and optional ReLU.
- [ ] Run all arithmetic tests and require zero byte mismatches.

## Task 4: Implement the parameterized INT8 convolution core

**Files:**
- Create: fpga/rtl/cnn/mac_array_i8.sv
- Create: fpga/rtl/cnn/conv2d_i8.sv
- Create: fpga/rtl/cnn/feature_tile_buffer.sv
- Create: fpga/rtl/cnn/weight_tile_buffer.sv
- Create: fpga/rtl/cnn/accumulator_buffer.sv
- Create: fpga/sim/conv2d_i8_tb.sv

Start with a functional configuration PIN=1, POUT=1, TILE_H=4 and TILE_W=4. Keep PIN, POUT, TILE_H and TILE_W as parameters so resource sweeps can later test 2/2, 4/4 and 8/8.

The control loop is output-channel tile, output-spatial tile, input-channel tile, kernel-y, kernel-x, MAC, bias, requant, optional ReLU and output write.

Support exactly these model forms:

    6x6 stride 2 padding 2
    3x3 stride 1 or 2 padding 1
    1x1 stride 1 padding 0

Padding uses signed zero for convolution. The core must verify the output shape using the standard convolution formula before execution.

Test in this order:
- model.0.conv for 6x6 and Cin=1.
- model.1.conv for 3x3 stride 2.
- one C3 internal 3x3 convolution.
- one node with Cin=512.
- model.24.m.0, model.24.m.1 and model.24.m.2 for linear Detect outputs.

- [ ] Write small deterministic tests for 1x1, 3x3, 6x6, padding, stride, negative values and accumulator overflow boundaries.
- [ ] Run the local-memory convolution testbench and verify failure.
- [ ] Implement the scalar functional path with the shared requant/ReLU units.
- [ ] Run tests and compare every output byte with generated Golden files.

## Task 5: Implement SPPF MaxPool

**Files:**
- Create: fpga/rtl/cnn/maxpool_i8.sv
- Create: fpga/sim/maxpool_i8_tb.sv

Support only kernel 5x5, stride 1 and padding 2. Padding value is -127. The unit must operate on signed INT8 and must not perform ReLU.

Tests cover all-negative tensors, corners, edges, center values, and all three serial SPPF MaxPool calls. Compare each intermediate output, not only the final SPPF tensor.

- [ ] Write the directed MaxPool tests.
- [ ] Run the testbench and verify failure.
- [ ] Implement line buffers and a signed 5x5 maximum window.
- [ ] Compare the three package MaxPool outputs byte-for-byte.

## Task 6: Implement Avalon-MM DMA and DDR behavior model

**Files:**
- Create: fpga/rtl/avalon/avalon_burst_reader.sv
- Create: fpga/rtl/avalon/avalon_burst_writer.sv
- Create: fpga/rtl/cnn/tensor_dma.sv
- Create: fpga/sim/avalon_ddr_model.sv
- Create: fpga/sim/tensor_dma_tb.sv

Use 64-byte bursts for feature, weight and output traffic. Bias and multiplier tables must support 4-byte alignment; shift tables support byte addressing.

When waitrequest is asserted, preserve address, burstcount, write data and byteenable. Store completion is asserted only after the final output burst is accepted. Any timeout, read error, write error or byte-count mismatch aborts the task.

Tests cover random waitrequest insertion, burst boundaries, tail transfers shorter than 64 bytes, unaligned rejection, read timeout, write timeout and concurrent read/write exclusion.

- [ ] Write the DDR model and DMA protocol tests.
- [ ] Run the DMA testbench and verify failure.
- [ ] Implement burst reader, burst writer and byte counters.
- [ ] Run tests with randomized waitrequest and require exact memory image comparison.

## Task 7: Implement Job Controller and unified operator executor

**Files:**
- Create: fpga/rtl/avalon/cnn_job_controller.sv
- Create: fpga/rtl/cnn/fpga_op_executor.sv
- Create: fpga/sim/cnn_job_controller_tb.sv
- Create: fpga/sim/fpga_op_executor_tb.sv

Controller states:

    IDLE -> VALIDATE -> LOAD_PARAMS -> LOAD_TILE -> EXECUTE
         -> STORE_TILE -> NEXT_TILE -> DONE
    Any error -> ERROR -> wait for CLEAR

Dispatch rules:

    CONV_RELU   -> conv2d_i8 + requant + relu
    CONV_LINEAR -> conv2d_i8 + requant
    MAXPOOL     -> maxpool_i8

Tests run CONV_RELU -> MAXPOOL -> CONV_LINEAR in one DDR model, verify task/frame IDs and IRQ order, reject duplicate START, force a watchdog timeout, and prove no remaining output is written after ERROR.

- [ ] Write the sequenced job tests and error-injection cases.
- [ ] Run the executor testbench and verify failure.
- [ ] Implement validation, dispatch, tile iteration, completion and error recovery.
- [ ] Run the testbench and compare output memory plus status registers.

## Task 8: Implement the external uint8-to-int8 model-input boundary

**Files:**
- Create: fpga/rtl/preprocess/model_input_u8_to_i8.sv
- Create: fpga/sim/model_input_u8_to_i8_tb.sv

The exported external model input is uint8, while model.0.conv consumes signed INT8. For the package input scale 1/127, implement:

    q_i8 = round_half_to_even(gray_u8 * 127 / 255)

The testbench reads all 65,536 bytes from golden_input_u8.bin and compares the converted result with vectors/model_0_conv_input_i8.bin. The intended data path is preprocessing DMA writer, uint8-to-int8 converter, INT8 model-input buffer and model.0 Conv.

- [ ] Write the full-frame conversion test.
- [ ] Run it and verify failure.
- [ ] Implement the exact integer conversion and valid/ready backpressure.
- [ ] Require all 65,536 converted bytes to match.

## Task 9: Establish smoke and full-package RTL regression

**Files:**
- Create: fpga/sim/run_smoke.ps1
- Create: fpga/sim/run_package_regression.ps1
- Create: fpga/sim/report_package_regression.py

Smoke regression runs CSR, Requant, ReLU, small 1x1 and 3x3 Conv, 6x6 first-layer Conv, MaxPool, DMA waitrequest and Job Controller tests on every RTL change.

Package regression runs all 60 convolution nodes, all 3 MaxPool nodes and all 3 Detect heads. Each case checks output length, CRC32, mismatch count and first mismatch location. The acceptance condition is mismatch_count=0 for every node.

ModelSim is currently available as ModelSim 10.6e. Use the shared runner rather than hard-coding a developer-specific work directory:

    powershell -ExecutionPolicy Bypass -File fpga/sim/run_msim.ps1 -Testbench fpga_op_executor_tb

- [ ] Write smoke and report-parser tests.
- [ ] Implement deterministic compile order and generated-vector discovery.
- [ ] Run smoke regression.
- [ ] Run the full package regression after the executor is stable and archive the report with the package SHA.

## Task 10: Quartus integration, resource sweep and G3 acceptance

**Files:**
- Create: fpga/quartus/ac550_soc.qpf
- Create: fpga/quartus/ac550_soc.qsf
- Create: fpga/constraints/ac550_soc.sdc
- Create: fpga/platform_designer/ac550_soc_system.qsys
- Create: fpga/reports/cnn_resource_sweep.csv
- Create: fpga/reports/cnn_g3_acceptance.md

Target Cyclone V SoC device is 5CSXFC6C6U23I7 with an initial 100 MHz CNN clock. Sweep PIN/POUT configurations 1/1, 2/2, 4/4 and 8/8 with 4x4 and 8x8 tiles.

Select the fastest configuration that has non-negative TimeQuest worst slack, no unconstrained clocks and no more than 80 percent use of DSP, ALM or M10K resources. Leave at least 20 percent resource headroom for camera, preprocessing and SoC interfaces.

If a build fails resource or timing constraints, reduce POUT, then PIN, then tile size. Do not change quantization, rounding, storage layout or graph partition.

G3 acceptance requires:
- All 60 Conv outputs bit-exact with the package.
- All 3 MaxPool outputs bit-exact with the package.
- Detect Linear outputs preserve negative logits.
- DDR byte counts, task IDs, frame IDs and IRQs are correct.
- Resource, timing and clock-constraint checks pass.
- HPS submits one real model Conv task and reads back the expected output.

## Milestone Gates

Milestone 1: model.0.conv, including 6x6 first-layer arithmetic, passes bit-exact RTL comparison.

Milestone 2: CONV_RELU -> MAXPOOL -> CONV_LINEAR runs through the DDR behavior model with correct descriptor, DMA, status and IRQ behavior.

Milestone 3: all 60 Conv nodes and 3 MaxPool nodes pass the package regression with zero byte mismatches.

Milestone 4: Quartus timing/resource reports pass and HPS can submit a real descriptor.

## Implementation Order

    package compiler
    -> descriptor and CSR
    -> requant and ReLU
    -> local-memory Conv
    -> MaxPool
    -> DMA
    -> Job Controller
    -> complete package regression
    -> Quartus synthesis and parallelism sweep
    -> HPS single-task integration

This document records the RTL implementation plan only. It does not mark the current INT8 package as G2-passed and does not modify the model package.
