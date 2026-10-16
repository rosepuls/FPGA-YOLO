# FPGA 目录

这里放 AC550-SoC 的 RTL、Quartus/Platform Designer 和仿真工程。

建议开发顺序：

1. 先用 Golden Model 生成输入、权重和输出定点向量；
2. 验证 `fpga/rtl/cnn` 的单层 Conv/ReLU/Requant；
3. 再连接 AXI 和 DMA；
4. 最后接入视频前端、Frame Buffer 和 ARM 后处理。

## INT8 算子任务链 bring-up

当前已具备 CSR、Conv/Requant/ReLU、MaxPool、任务控制器、Avalon-MM burst DMA，以及功能级 `fpga_op_executor`。DMA 使用 32-bit words 计数，burst 最多 16 words（64 bytes），在共享 master 上串行执行读/写；功能特定区域要求 64-byte 对齐，bias/multiplier 要求 4-byte 对齐，shift 支持 byte 地址。DMA 超时、错误和尾部 byte-enable 有独立仿真覆盖。可用 ModelSim 运行对应测试：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File fpga/sim/run_msim.ps1 -Testbench cnn_job_controller_tb
powershell -NoProfile -ExecutionPolicy Bypass -File fpga/sim/run_msim.ps1 -Testbench tensor_dma_tb
powershell -NoProfile -ExecutionPolicy Bypass -File fpga/sim/run_msim.ps1 -Testbench fpga_op_executor_tb
```

`fpga_op_executor` 目前把完整输入、权重和输出 tensor 放在本地数组中，目的是验证 DDR→算子→DDR 的功能闭环。它尚未实现全模型可综合的片上 tile 复用结构，也未完成 Quartus 资源/时序验收；整网部署前必须把这些完整 tensor bank 改为有界 feature/weight tile buffer。

DMA 读命令和写 beat 共用一个状态机与 Avalon master，避免对同一物理接口建立并行访问通路；`fpga_op_executor.master_burstcount` 已将 burst 长度带到系统边界。`avalon_ddr_model` 为测试用确定性 stall/错误注入模型，不是板级 DDR 控制器。

## AC550 package RTL 回归

先编译当前冻结模型包，再选择一个或多个节点用 ModelSim 跑 RTL；省略 `--node` 时遍历完整 63 个 FPGA descriptor：

```powershell
$env:AC550_PACKAGE_ROOT = 'runs/ac550_yolov5n_relu_int8_qat_staged_v1'
python fpga/tools/make_ac550_rtl_vectors.py --package $env:AC550_PACKAGE_ROOT --output fpga/sim/generated
python fpga/tools/run_ac550_rtl_regression.py --package $env:AC550_PACKAGE_ROOT --node model.24.m.0
```

回归逐节点记录 RTL 输出 CRC、首个不匹配，并与由导出权重/bias/quant 参数构造的独立整数参考对拍。它还单独报告原包 tensor Golden 与静态 package scale 的差别：当前导出器的 tensor Golden 按每张样本自身 max-abs/127 编码，卷积参数则按固定 QAT scale，因此不能直接把两套 INT8 编码视为同一数值格式。全量 RTL 回归会受当前标量 MAC 核的周期数影响。

有界 tile 的第一批基础模块已新增：`feature_tile_buffer`、`weight_tile_buffer`、`accumulator_buffer` 和参数化 `mac_array_i8`。运行小 tile 的独立检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File fpga/sim/run_msim.ps1 -Testbench tile_datapath_tb
```

这些 tile building blocks 目前还没有接入 `conv2d_i8` / `fpga_op_executor`；DDR tile 调度、卷积 tile 控制，以及替换完整 tensor banks 仍是下一项集成工作。
