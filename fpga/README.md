# FPGA 目录

这里放 AC550-SoC 的 RTL、Quartus/Platform Designer 和仿真工程。

建议开发顺序：

1. 先用 Golden Model 生成输入、权重和输出定点向量；
2. 验证 `fpga/rtl/cnn` 的单层 Conv/ReLU/Requant；
3. 再连接 Avalon-MM 和 DMA；
4. 最后接入视频前端、Frame Buffer 和 ARM 后处理。

## INT8 算子任务链 bring-up

当前已具备 CSR、Conv/Requant/ReLU、MaxPool、任务控制器、有界 tile 执行器及 Avalon-MM burst DMA。描述符的 tensor 基地址要求 64-byte 对齐，执行器发起的 tile 行和 OIHW 权重切片可从任意字节地址开始；DMA 在 32-bit 总线上发出最多 16 words（64 bytes）的 burst。超时、错误、非对齐头尾包和 byte-enable 有独立仿真覆盖。可用 ModelSim 运行对应测试：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File fpga/sim/run_msim.ps1 -Testbench cnn_job_controller_tb
powershell -NoProfile -ExecutionPolicy Bypass -File fpga/sim/run_msim.ps1 -Testbench tensor_dma_tb
powershell -NoProfile -ExecutionPolicy Bypass -File fpga/sim/run_msim.ps1 -Testbench fpga_op_executor_tb
```

`fpga_op_executor` 以 8×8 通道、4×4 输出空间 tile 从 DDR 读取输入/权重并写回输出，不再缓存完整 tensor；`ac550_cnn_accelerator` 是 CSR 到执行器的可综合顶层连线。当前完成的是 ModelSim 行为级验收，尚未做 Quartus 资源/时序或板级验收。

DMA 读命令和写 beat 共用一个状态机与 Avalon master，避免对同一物理接口建立并行访问通路；`fpga_op_executor.master_burstcount` 已将 burst 长度带到系统边界。`avalon_ddr_model` 为测试用确定性 stall/错误注入模型，不是板级 DDR 控制器。

## AC550 package RTL 回归

以 v4 冻结模型包编译向量并运行 ModelSim；省略 `--node` 时遍历完整 63 个 FPGA descriptor：

```powershell
$env:AC550_PACKAGE_ROOT = 'runs/ac550_yolov5n_relu_int8_qat_staged_v4'
python fpga/tools/make_ac550_rtl_vectors.py --package $env:AC550_PACKAGE_ROOT --output fpga/sim/results/v4-vectors
python fpga/tools/run_ac550_ddr_regression.py --package $env:AC550_PACKAGE_ROOT --vectors fpga/sim/results/v4-vectors --output fpga/sim/results/v4-ddr
```

回归逐节点记录 DDR 输出 CRC、首个不匹配、模拟周期数和读写请求数，同时与 v4 Golden 和独立整数参考对拍。当前 63 个 FPGA 节点、1,953,216 字节全部零差异；详见 `fpga/reports/ac550-int8-package-ddr-regression-v4.md`。

有界 tile 的第一批基础模块已新增：`feature_tile_buffer`、`weight_tile_buffer`、`accumulator_buffer` 和参数化 `mac_array_i8`。运行小 tile 的独立检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File fpga/sim/run_msim.ps1 -Testbench tile_datapath_tb
```

`conv2d_i8` 的计算 tile 和 `fpga_op_executor` 的 DDR streaming 均已接入回归。后续重点是 Quartus RAM 推断、资源与时序收敛，以及 Platform Designer/HPS 和板级链路。
