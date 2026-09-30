# FPGA 目录

这里放 ZYNQ-7010 的 RTL、Vivado、Vitis 和仿真工程。

建议开发顺序：

1. 先用 Golden Model 生成输入、权重和输出定点向量；
2. 验证 `fpga/rtl/cnn` 的单层 Conv/ReLU/Requant；
3. 再连接 AXI 和 DMA；
4. 最后接入视频前端、Frame Buffer 和 ARM 后处理。
