# FPGA-YOLO 项目结构

本项目按“数据 → PyTorch 参考模型 → INT8/Golden Model → FPGA CNN → PS/PL 联调”组织。

```text
FPGA-YOLO/
├─ data/                            # 本地数据集（已加入 .gitignore）
│  ├─ GC10-DET/                     # 原始 GC10 数据
│  ├─ GC10-DET-COCO/                # 审计后的 COCO 中间集
│  └─ GC10-DET-YOLO-ROI256/         # 256×256 灰度 YOLO 训练集
├─ configs/
│  ├─ yolov5n_gray_256.yaml          # 10 类 YOLOv5n 单通道模型结构
│  └─ yolov5n_gray_silu_baseline.yaml# SiLU 基线训练参数
├─ models/
│  ├─ gc10_coco/                    # GC10 数据审计与 COCO 转换工具
│  ├─ convert_gc10_to_coco.py       # 原始 GC10 → COCO
│  ├─ convert_coco_to_yolo_roi256.py# COCO → YOLO 256×256 灰度
│  └─ tests/                        # 数据转换测试
├─ third_party/
│  ├─ yolov5/                       # YOLOv5 v7.0 灰度训练工作副本
│  └─ yolov5-v7.0-original/         # 上游源码不可变本地备份（Git 忽略）
├─ fpga/
│  ├─ rtl/cnn/                      # Conv Engine、PE、Feature/Weight Buffer
│  ├─ rtl/common/                   # FIFO、计数器、定点辅助模块
│  ├─ rtl/axi/                      # AXI-Lite、AXI-Stream、DMA 接口
│  ├─ sim/                          # RTL 仿真 testbench 和向量
│  ├─ vivado/                       # Vivado 工程/Tcl/Block Design
│  ├─ vitis/                        # Vitis 裸机或 Linux 应用工程
│  └─ constraints/                  # XDC 时钟、引脚和时序约束
├─ ps_app/
│  ├─ src/                          # ARM 端 Decode、NMS、PASS/FAIL
│  ├─ include/                      # ARM 端头文件
│  └─ build/                        # PS 编译产物（本地生成）
├─ tools/
│  └─ train_yolov5n_silu.py         # 预检通过后启动 SiLU 基线训练
├─ runs/                            # 权重、指标和图表（运行时生成，Git 忽略）
└─ docs/                            # 设计文档和执行计划
```

当前数据入口：

```text
data/GC10-DET-YOLO-ROI256
```

当前模型路线：

```text
GC10-DET（10 类）
→ YOLOv5n 单通道 SiLU 精度基线
→ LeakyReLU 部署候选
→ INT8 Golden Model
→ YOLOv5n FPGA Accelerator
→ ARM Decode/NMS
```
