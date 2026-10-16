# AC550 bounded tile DDR executor design

## Scope

Keep the existing CSR descriptor format, task/frame completion, IRQ, alignment checks, watchdog, and one-outstanding-DMA contract. Replace the executor's full-tensor staging arrays with fixed-size feature, weight, accumulator, quantization, and output tile banks. The v4 arithmetic baseline remains a separate oracle; a package-integrated CSR/DDR test must compare DDR bytes to its Golden vectors.

## Dataflow

- Conv uses an 8-input-channel × 8-output-channel × 4×4 output tile. The largest 6×6, stride-2 input footprint is 12×12 per input lane; fixed local banks contain at most 1,152 feature bytes, 2,304 weight bytes, 512 accumulator bytes, 72 quantization bytes, and 128 output bytes. No bank grows with image dimensions or total channel count.
- For each output-channel/spatial tile, clear accumulators, fetch bias/multiplier/shift for the eight output channels, then iterate input-channel blocks. Fetch only valid rows of each feature footprint and each OIHW weight kernel. Padding samples contribute zero. The existing 8×8 MAC and requant semantics apply after all input-channel blocks; ReLU is applied only for `OP_CONV_RELU`.
- MaxPool uses a 4×4 output tile and a 5×5 sliding window over at most an 8×8 input footprint. Out-of-image samples contribute signed -128. Each output row is written back after the tile computes.
- Each DDR subtransfer uses a byte address, byte count, and local byte offset. Descriptor tensor bases remain 64-byte aligned; row and kernel-slice addresses may be unaligned. DMA must preserve neighboring bytes on masked first/last words and stop after an error.
- Keep a single outstanding request. The controller validates and latches the descriptor, starts the streaming executor, and records completion/error. The CSR exposes the complete latched descriptor through the synthesizable `ac550_cnn_accelerator` wrapper; high bits that cannot fit the executor's shape fields are rejected before START. Complete tensor ranges must fit 32-bit DDR addresses and output must not overlap any source tensor. The executor owns subtransfers and signals done only after the final DDR output write completes. On watchdog timeout it blocks new tile requests, drains the current DMA transaction, then reports error.

## Acceptance order

1. DMA unaligned feature/output rows, partial tails, and neighbor-byte preservation.
2. Small CSR/DDR jobs: padded Conv, 5×5 MaxPool edges, linear Conv, multi-tile/channel, write abort, and task/IRQ ordering.
3. v4 `model.0` 6×6 Conv, one SPPF MaxPool, one Detect linear Conv, then all 63 FPGA nodes. Record per-node byte counts and exact mismatch locations. Do not claim full-link acceptance based on the core-only 63-node baseline.

## Risks

The straightforward channel-block scheduler reloads feature footprints for each output-channel block and spatial tile. It is functionally bounded but bandwidth-heavy. Benchmark before Quartus closure and add reuse or a line cache only if needed; such optimization must not change the Golden contract.
