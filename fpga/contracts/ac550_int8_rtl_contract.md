# AC550 INT8 RTL Runtime Contract

## Package identity

The vector compiler takes an explicit `--package` path, computes the SHA-256 of `manifest.json`, and records it in `fpga/sim/generated/package_lock.json`. A later generation in the same output directory fails when this identity differs, preventing Golden vectors, runtime tables and RTL results from being mixed between exports.

The engineering baseline is `runs/ac550_yolov5n_relu_int8_qat_staged_v1`; its manifest SHA-256 is `3BC1316D55C85B9010052E4ABCD1F886B74146DB7314CE2475EE6E45D016EE24`. Its state is `PRE_G2_STATIC_QUANTIZATION`, so it is for bring-up rather than final G2 accuracy release.

## Memory representation

All generated `.mem` images are byte-addressed hexadecimal: one two-digit byte per line in original file order. INT8 keeps its two's-complement bit pattern. INT32 bias remains little-endian as stored by `bias_i32.bin` and must be assembled as little-endian words.

`conv_layer_table.bin` holds 64-byte little-endian records. Bytes 0-3 are descriptor index; byte 4 is opcode (Conv+ReLU=1, Conv Linear=2, MaxPool=3); bytes 5-7 are square kernel, stride and symmetric padding; bytes 8-19 are `uint16` Cin, Hin, Win, Cout, Hout, Wout; bytes 20-43 are `uint32` weight offset/bytes, bias offset/bytes, multiplier offset, shift offset; bytes 44-45 are requant channel count (zero for MaxPool); bytes 46-63 are reserved zero. `requant_multiplier_i32.bin` concatenates signed little-endian per-channel INT32 multipliers and `requant_shift_u8.bin` has its unsigned shifts; descriptor offsets are bytes.

## Graph-name normalization

Graph nodes such as `model.0` and `model.2.cv1` resolve to quantization and Golden names ending in `.conv`. Detect nodes `model.24.m.0`, `model.24.m.1` and `model.24.m.2` remain unsuffixed and use the linear opcode. The serialized SPPF calls map `model.9.maxpool.1`, `.2`, `.3` to `model.9.m`, `model.9.m#2`, `model.9.m#3`.

## Safety checks

The compiler requires 87 graph nodes, 60 convolutions and executor split 57/3/3/7/13/4. It validates every Golden input/output mapping and every weight or bias offset plus byte count against the source binary before emitting the associated `.mem` image.

## Task and DMA execution order

`cnn_job_controller` validates the descriptor and local buffer capacities before issuing DMA. A Conv task reads regions in this order: input, weight, bias, multiplier, shift; then starts the selected Conv mode and writes output. A MaxPool task reads input, starts MaxPool, and writes output. Every DMA response must report the requested byte count before execution can advance. A read/write failure, short transfer, engine failure, or watchdog timeout ends the task with an error and does not signal successful completion.

The Avalon-MM master moves 32-bit little-endian words. DDR addresses are word aligned; local addresses are byte offsets within the selected region. The final partial write uses `byteenable` so bytes beyond the requested tensor are preserved. Region IDs are input=0, weight=1, bias=2, multiplier=3, shift=4, output=5.

`fpga_op_executor` is the functional integration harness for DDR→Conv/MaxPool→DDR simulation. It currently holds full tensors in local arrays. It has not passed Quartus fitting and is not the final resource-bounded AC550 implementation; replace its full tensor banks with reusable feature and weight tiles before full-model deployment.
