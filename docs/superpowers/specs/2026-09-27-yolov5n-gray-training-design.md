# YOLOv5n 单通道 SiLU 训练适配设计

## 目的

为 ZYNQ-7010 的首轮软件精度基线建立可复现的 YOLOv5n 训练路径。输入、
训练数据、验证数据与后续 FPGA 接口均固定为 `256×256×1` 灰度图；类别数固定
为 10。首轮模型保留 YOLOv5 的 SiLU 激活，只用于衡量软件精度上限。

## 上游代码与备份边界

- 工作副本：`third_party/yolov5`（YOLOv5 v7.0）。
- 原始备份：`third_party/yolov5-v7.0-original`。训练适配只改工作副本，备份
  永不修改。
- 项目自有配置、启动命令和测试不放入上游目录；它们放在项目根目录的
  `configs/` 与 `tools/` 下。

## 单通道适配

训练和验证入口新增 `--channels`，首轮固定为 `--channels 1`。该参数必须同时
传给模型构造和训练/验证数据加载器。

数据加载器以 OpenCV 灰度模式读取 PNG，并在内部维持 `H×W×1`，使 Mosaic、
几何增强、letterbox 和批处理仍使用明确的通道维。张量输出固定为
`B×1×H×W`。仅 RGB 专用的 HSV 增强在单通道模式下跳过；Mosaic、缩放、翻转和
仿射等几何增强保留。训练首轮不启用缓存，避免旧的三通道缓存文件混入结果。

模型由 YOLOv5n YAML 构造，类别数由 `gc10_yolo.yaml` 的 `nc: 10` 覆盖。输入
为 `256` 时，三尺度检测头的原始卷积通道均为 `3×(5+10)=45`，对应空间尺寸
为 `32×32`、`16×16` 和 `8×8`。

## 预训练权重初始化

首轮训练以官方 `yolov5n.pt` 为初始权重。加载 checkpoint 时：

1. 若目标模型为单通道、checkpoint 首层卷积为三通道，取首层权重在输入通道
   维度的算术均值，形状从 `[out, 3, k, k]` 转为 `[out, 1, k, k]`。
2. 用转换后的首层权重和其余所有形状匹配的参数初始化单通道模型。
3. 检测头因类别数从 COCO 的 80 改为 10 而不加载，按 YOLOv5 默认初始化。
4. 日志必须明确记录已转换第一层及成功转移的参数数。

该转换只在运行时对内存中的 checkpoint state dict 执行；不覆盖官方
`yolov5n.pt`。

## 配置与训练入口

新增 `configs/yolov5n_gray_256.yaml`，保持 v7.0 YOLOv5n 的深度、宽度、Anchor
与三尺度结构，只将默认类别数设为 10，并标注它必须以单通道模型入口构造。

新增一个项目脚本作为首轮基线的唯一入口。它固定：

- 数据配置：`gc10_yolo.yaml`；
- 模型配置：`configs/yolov5n_gray_256.yaml`；
- 输入尺寸：256；
- 通道：1；
- 权重：官方 `yolov5n.pt`；
- 激活：上游 YOLOv5 的 SiLU；
- epoch：300；batch size：32；seed：20260927；
- 输出目录：`runs/yolov5n_gray256_silu_baseline/`。

YOLOv5 自动保存的 `opt.yaml`、`hyp.yaml`、loss/指标曲线、混淆矩阵与
`best.pt`/`last.pt` 是该基线的训练记录。`runs/` 加入 `.gitignore`。

## 预检与验收

在启动长训练前，自动检查：

1. 数据 YAML 指向 `data/GC10-DET-YOLO-ROI256`，`nc` 为 10；
2. 一个数据加载批次的形状为 `B×1×256×256`；
3. 模型第一层卷积的输入通道为 1；
4. 三个检测头输出卷积通道均为 45；
5. RGB 预训练首层的均值转换与加载后权重数值一致。

预检失败时不得启动 epoch 训练。完成首轮训练后，使用同一单通道验证加载器
评估验证集和测试集，保存总体及逐类别指标，作为后续 LeakyReLU 对照的唯一
基线。

## 非目标

本阶段不实现 LeakyReLU、INT8 量化、RTL、PS Decode/NMS，也不修改数据集或
`third_party/yolov5-v7.0-original` 备份。
