# YOLOv5n-ReLU 四分类灰度基线训练指令

> 目的：在小梅哥 AC550 的 FPGA 部署约束下，训练与 YOLOv5n 拓扑一致、普通卷积使用 ReLU、输入为单通道灰度的四分类模型。本模型是后续 INT8 量化、FPGA Conv/ReLU/MaxPool 加速和 ARM 后处理的候选模型。

本流程使用独立源码副本：

```text
C:\Study\IC_Study\Project\FPGA\Contest\CGEDC_1\FPGA-YOLO\third_party\yolov5-relu
```

原始 `third_party\yolov5` 保留为 SiLU 基线，不要在其中训练或修改 ReLU 模型。

## 固定训练契约

| 项目 | 固定值 |
| --- | --- |
| 平台 | 小梅哥 AC550，Cyclone V SoC |
| 网络 | YOLOv5n，保持原始深度/宽度系数 |
| 普通卷积激活 | ReLU |
| Detect 最后卷积 | 线性输出，不加 ReLU |
| 输入 | 单通道灰度，`1×256×256` |
| 类别数 | 4 |
| 类别顺序 | `ca_shang`、`zang_wu`、`zhe_zhou`、`zhen_kong` |
| 模型配置 | `third_party\yolov5-relu\models\yolov5n_relu_4cls_gray.yaml` |
| 数据配置 | `lv_yolo.yaml` |
| 训练输入 | `--imgsz 256 --channels 1` |
| 推荐 batch size | `32`，显存不足时依次降为 `16`、`8`、`4` |
| 随机种子 | `20260930` |

Detect 的每个输出头必须为 `3 × (5 + 4) = 27` 通道。输入为 `256×256` 时，三个训练态输出尺寸应为：

```text
[1, 3, 32, 32, 9]
[1, 3, 16, 16, 9]
[1, 3,  8,  8, 9]
```

## 1. 进入工程和训练环境

```powershell
Set-Location 'C:\Study\IC_Study\Project\FPGA\Contest\CGEDC_1\FPGA-YOLO'
conda activate zynq-defect
python --version
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果 `conda run -n zynq-defect` 报 `NoWritableEnvsDirError`，可以直接使用该环境解释器：

```powershell
$Py = 'D:\Conda\conda_envs\zynq-defect\python.exe'
& $Py -c "import torch; print(torch.__version__)"
```

下面示例默认已经执行 `conda activate zynq-defect`；如果使用 `$Py`，将示例中的 `python` 替换为 `& $Py`。

## 2. 训练前契约检查

先确认 YAML、单通道数据加载和 ReLU 模型契约：

```powershell
python -m unittest `
  models.tests.test_yolov5_relu_variant `
  models.tests.test_yolov5_gray_dataloader `
  -v
```

检查必须满足：

- 首层输入通道为 `1`。
- 普通 `Conv` 模块均为 ReLU，没有 SiLU。
- Detect 三个末端卷积均为线性 `Conv2d`，输出通道为 `27`。
- `data\LV-YOLO-ROI256\images\train`、`val` 和 `test` 均存在。
- 输入图像为 `256×256` 单通道灰度 PNG。

契约测试失败时不要开始训练。

## 3. 使用 GPU 训练 ReLU 模型

为了与已有 SiLU 基线公平比较，第一轮使用官方 `yolov5n.pt` 作为迁移学习初始化，并保持相同的数据、输入尺寸、随机种子和训练轮数。

```powershell
Set-Location 'C:\Study\IC_Study\Project\FPGA\Contest\CGEDC_1\FPGA-YOLO\third_party\yolov5-relu'

python train.py `
  --weights '..\..\yolov5n.pt' `
  --cfg '.\models\yolov5n_relu_4cls_gray.yaml' `
  --data '..\..\lv_yolo.yaml' `
  --hyp '.\data\hyps\hyp.scratch-low.yaml' `
  --imgsz 256 `
  --channels 1 `
  --epochs 100 `
  --batch-size 32 `
  --device 0 `
  --workers 4 `
  --seed 20260930 `
  --project '..\..\runs' `
  --name 'yolov5n_gray_relu_256'
```

不要使用 `--batch-size -1`。当前灰度通道适配与 YOLOv5 AutoBatch 探测存在不兼容风险；显存不足时只降低 batch size，不要修改 `--imgsz` 或 `--channels`。

如果希望在第二轮实验中使用已经训练好的 SiLU 缺陷权重作为 warm start，可以单独记录该实验，并将 `--weights` 改为：

```powershell
'..\..\runs\yolov5n_gray_silu_baseline2\weights\best.pt'
```

该实验不能替代使用官方 `yolov5n.pt` 的公平基线，两个实验必须使用不同的 run name。

## 4. CPU 流程验证

只用于确认命令、配置和数据链路可运行，不用于最终精度结论：

```powershell
python train.py `
  --weights '..\..\yolov5n.pt' `
  --cfg '.\models\yolov5n_relu_4cls_gray.yaml' `
  --data '..\..\lv_yolo.yaml' `
  --hyp '.\data\hyps\hyp.scratch-low.yaml' `
  --imgsz 256 `
  --channels 1 `
  --epochs 5 `
  --batch-size 8 `
  --device cpu `
  --workers 0 `
  --seed 20260930 `
  --project '..\..\runs' `
  --name 'yolov5n_gray_relu_cpu_smoke'
```

## 5. 在独立 test split 上验证

训练完成后，仍在 `third_party\yolov5-relu` 目录执行：

```powershell
python val.py `
  --weights '..\..\runs\yolov5n_gray_relu_256\weights\best.pt' `
  --data '..\..\lv_yolo.yaml' `
  --imgsz 256 `
  --channels 1 `
  --task test `
  --device 0 `
  --batch-size 32 `
  --workers 0 `
  --project '..\..\runs' `
  --name 'yolov5n_gray_relu_256_test'
```

需要记录：

- 每类 Precision、Recall、mAP@0.5、mAP@0.5:0.95。
- 混淆矩阵。
- `zhe_zhou` 类的误检和漏检情况。
- 与 SiLU 基线在同一 test split 上的差异。

## 6. 保存模型和复现信息

回到工程根目录执行：

```powershell
Set-Location 'C:\Study\IC_Study\Project\FPGA\Contest\CGEDC_1\FPGA-YOLO'
Get-FileHash '.\runs\yolov5n_gray_relu_256\weights\best.pt' -Algorithm SHA256
```

必须保留：

```text
runs/yolov5n_gray_relu_256/weights/best.pt
runs/yolov5n_gray_relu_256/weights/last.pt
runs/yolov5n_gray_relu_256/results.csv
runs/yolov5n_gray_relu_256/results.png
runs/yolov5n_gray_relu_256/opt.yaml
runs/yolov5n_gray_relu_256/hyp.yaml
runs/yolov5n_gray_relu_256_test/
```

同时记录训练命令、数据集版本、随机种子、权重 SHA-256、YOLOv5-relu 目录版本和 test split 指标。

## 7. 部署前验收门槛

ReLU 训练结束后不能直接把 `best.pt` 写入 FPGA。必须先完成：

1. 使用与 AC550 相同的 RGB565→灰度、ROI 和 `256×256` Resize Golden 输入进行推理验证。
2. 融合 Conv+BN，并完成 INT8 权重/激活、INT32 bias/累加的量化评估。
3. 导出每个 Conv、ReLU、MaxPool、ARM Add/Concat/Upsample 节点的 Golden tensor。
4. 导出 P3/P4/P5 三路 INT8 检测头、anchors 和量化参数。
5. 确认 INT8 模型与 FP32 ReLU 模型的框、类别、置信度和 NMS 结果在规定容差内一致。

只有 ReLU 精度和 INT8 Golden 均通过后，才进入 AC550 FPGA 算子实现和 ARM 图调度联调。

## 常见错误

- 不要使用 `third_party\yolov5\models\yolov5n.yaml` 训练 ReLU 模型。
- 不要把现有 SiLU `best.pt` 当作最终 FPGA 权重；它只能作为对照或可选 warm start。
- 不要把 Detect 最后一层接 ReLU，否则会破坏检测 logits。
- 不要改成 RGB 三通道、`640×640` 或 letterbox 输入来绕过问题。
- 不要为了提高 FPS 删除 C3、SPPF 或检测头分支；AC550 资源优化应在量化、tile 和并行度阶段处理。
