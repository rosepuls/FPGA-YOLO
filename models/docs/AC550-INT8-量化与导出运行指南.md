# AC550 YOLOv5n-ReLU INT8 量化与导出

本指南运行项目的第一个 AC550 模型包导出器。它以冻结的 ReLU `best.pt` 和
`256×256` 单通道灰度校准图为输入，输出 FPGA/HPS 联调需要的 INT8 权重、INT32
bias、量化参数、执行图和第一张图的 Golden tensors。

## 运行前检查

在 `FPGA-YOLO` 根目录的 PowerShell 中运行：

```powershell
$py = 'D:\Conda\conda_envs\zynq-defect\python.exe'

& $py -m unittest `
  models.tests.test_yolov5_relu_variant `
  models.tests.test_yolov5n_relu_export -v
```

预期：6 个测试通过。测试会验证 ReLU 模型契约、INT8 对称量化规则、图分工和
端到端包导出。

## 生成 PTQ 模型包

以下命令选取训练集目录中按文件名排序的前 64 张 `L` 模式、`256×256` 图像作为
校准集。输出目录必须不存在；这是为了防止覆盖已冻结的模型包。

```powershell
$py = 'D:\Conda\conda_envs\zynq-defect\python.exe'

& $py .\models\export\export_yolov5n_relu_ac550.py `
  --weights .\runs\yolov5n_gray_relu_256\weights\best.pt `
  --relu-root .\third_party\yolov5-relu `
  --calibration-dir .\data\LV-YOLO-ROI256\images\train `
  --calibration-images 64 `
  --output .\runs\ac550_yolov5n_relu_int8
```

成功后命令会打印模型包目录和 `manifest.json` 的 SHA-256。将两者写入实验记录；
后续 FPGA、HPS 和 Golden tensor 对比必须使用同一个 manifest。

## 产物说明

```text
runs/ac550_yolov5n_relu_int8/
  manifest.json            # 版本、权重 SHA、输入契约、文件哈希和图节点数量
  graph.json               # FPGA/ARM 节点分工
  weights_i8.bin           # 按 graph 源模块顺序串接的 INT8 权重
  bias_i32.bin             # 对应的 INT32 bias
  quant_params.json        # Conv 的 scale/multiplier，以及校准得到的节点边界 scale
  anchors.json              # P3/P4/P5 stride 和像素 anchor
  golden_input_u8.bin      # 第一张校准图，长度固定为 65536 byte
  head_p3_i8.bin           # 第一张图的 P3 检测头 Golden tensor
  head_p4_i8.bin           # 第一张图的 P4 检测头 Golden tensor
  head_p5_i8.bin           # 第一张图的 P5 检测头 Golden tensor
  golden_vectors.json      # 每个记录边界 tensor 的形状、量化参数和文件名
  vectors/                  # Conv/Pool/Upsample/Concat/残差边界 Golden tensors
```

`graph.json` 的固定分工为：普通 Conv+ReLU、Detect 线性 Conv 和 SPPF 三次
MaxPool 在 FPGA；C3 残差 Add、Concat 与 FPN/PAN Upsample 在 ARM。Detect 末层
不会被加上 ReLU。

## 当前验收边界

导出器完成 Conv+BN 融合、权重 INT8 打包、bias INT32 计算、每层量化参数生成和
Golden tensor 导出。它是 G2 的输入，不代表 G2 已通过。

G2 还需要实现并运行 Python 的逐节点 INT8 图执行：使用相同输入/阈值对比 FP32
ReLU 模型与 INT8 图的逐层误差、P3/P4/P5、bbox、类别和置信度。只有该比较达标，
才冻结模型包并交给 FPGA Conv/ReLU/MaxPool RTL 实现。

## G2 多图验收

当前仓库提供批量验收入口。它会固定按文件名排序选择输入，复用同一份模型包和
manifest，对每张图记录 P3/P4/P5、检测框、类别、置信度，并汇总最坏值。默认门槛
是 head mean absolute error `0.25`、head RMSE `0.35`、匹配框 IoU `0.90`、
置信度差 `0.05`；这些是当前工程的预验收门槛，正式现场容差仍需结合数据集确认。

```powershell
$py = 'D:\Conda\conda_envs\zynq-defect\python.exe'

& $py .\models\export\ac550_g2_acceptance.py `
  --package .\runs\ac550_yolov5n_relu_int8 `
  --weights .\runs\yolov5n_gray_relu_256\weights\best.pt `
  --relu-root .\third_party\yolov5-relu `
  --image-dir .\data\LV-YOLO-ROI256\images\train `
  --max-images 16 `
  --min-images 16 `
  --output .\runs\ac550_yolov5n_relu_int8\g2_acceptance.json
```

退出码为 `0` 才表示当前门槛通过；退出码为 `2` 表示报告已生成但验收失败。
`g2_acceptance.json` 中的 `failure_reasons` 是阻塞原因。只要
`integer_simulation_complete` 为 `false`，就不能冻结 manifest 或宣称 FPGA G2
通过；当前版本的 Conv 节点已使用包内 INT8 参数，ARM Add/Concat/Upsample 和
FPGA MaxPool 的整数执行仍需补齐。
