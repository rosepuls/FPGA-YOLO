# YOLOv5n SiLU 四分类灰度基线训练指令

> 目的：先训练一个未替换激活函数的 YOLOv5n（默认 SiLU）基线模型，验证 LV 缺陷数据、四分类标注和 `256×256` 单通道输入的可达精度。本轮结果是后续 ReLU 模型和 INT8 量化的对照基线，不是最终 FPGA 部署模型。

## 固定训练契约

| 项目 | 固定值 |
| --- | --- |
| 网络 | YOLOv5n，默认 SiLU 激活 |
| 输入 | 单通道灰度，`1×256×256` |
| 类别数 | 4 |
| 类别顺序 | `ca_shang`、`zang_wu`、`zhe_zhou`、`zhen_kong` |
| 数据配置 | `C:\Study\IC_Study\Project\FPGA\Contest\CGEDC_1\FPGA-YOLO\lv_yolo.yaml` |
| 预训练权重 | `yolov5n.pt`；仅作为迁移学习初始化 |
| 随机种子 | `20260930` |

不要用 RGB 三通道输入、COCO 80 类数据配置、letterbox 以外的训练预处理规则来替代本基线。后续板端输入固定为灰度 `256×256` 直接 Resize，因此本次训练也固定使用 `--imgsz 256 --channels 1`。

## 1. 打开环境并进入工程

在 PowerShell 中执行：

```powershell
conda activate zynq-defect
Set-Location 'C:\Study\IC_Study\Project\FPGA\Contest\CGEDC_1\FPGA-YOLO'
python --version
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

若 `conda activate` 不可用，先执行一次 `conda init powershell`，关闭并重新打开 PowerShell 后再运行上述命令。

## 2. 训练前检查

确认数据集配置仍为四类并且转换、灰度数据加载测试通过：

```powershell
Get-Content '.\lv_yolo.yaml'
python -m unittest models.tests.test_lv_yolo_roi256 models.tests.test_yolov5_gray_dataloader -v
```

检查必须满足：

- `lv_yolo.yaml` 中为 `nc: 4`；类别顺序与本文“固定训练契约”完全一致。
- 测试通过；失败时先修正数据集或灰度数据加载，不启动训练。
- `data\LV-YOLO-ROI256\images\train`、`val` 和 `test` 均存在。

## 3. GPU 训练（推荐）

只有上一节输出 `cuda=True` 时使用本命令。该命令使用 YOLOv5 自带的 `models/yolov5n.yaml`，其默认激活函数为 SiLU；`--data` 中的四分类定义会覆盖 YAML 内的默认类别数。

```powershell
Set-Location '.\third_party\yolov5'

python train.py `
  --weights '..\..\yolov5n.pt' `
  --cfg '.\models\yolov5n.yaml' `
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
  --name 'yolov5n_gray_silu_baseline'
```

当前 YOLOv5 的 AutoBatch 会固定创建三通道探测张量，不能用于本项目的
`--channels 1` 模型，因此**不要使用** `--batch-size -1`。本机 RTX 5070 Laptop
GPU（约 12 GB 显存）先使用固定的 `--batch-size 32`；若出现 CUDA out of memory，依次改为
`16`、`8` 或 `4`。不要修改输入尺寸或通道数来绕过显存问题。

## 4. CPU 训练（仅用于流程验证）

CPU 训练通常很慢，适合验证命令、数据和标签是否可用，不能作为正式的精度结论。先将工作目录保持在 `third_party\yolov5`，再执行：

```powershell
python train.py `
  --weights '..\..\yolov5n.pt' `
  --cfg '.\models\yolov5n.yaml' `
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
  --name 'yolov5n_gray_silu_cpu_smoke'
```

CPU 流程成功后，切换到 GPU 并重新执行第 3 节的正式训练；正式训练目录必须使用新的、未存在的 `--name`。

## 5. 验证最佳权重

GPU 训练完成后，在 `third_party\yolov5` 目录执行：

```powershell
python val.py `
  --weights '..\..\runs\yolov5n_gray_silu_baseline\weights\best.pt' `
  --data '..\..\lv_yolo.yaml' `
  --imgsz 256 `
  --channels 1 `
  --task test `
  --device 0 `
  --project '..\..\runs' `
  --name 'yolov5n_gray_silu_baseline_test'
```

如果没有 GPU，将 `--device 0` 改为 `--device cpu`。

## 6. 必须保存的结果

本轮训练结束后，保留并记录下列内容：

```text
runs/yolov5n_gray_silu_baseline/weights/best.pt
runs/yolov5n_gray_silu_baseline/results.csv
runs/yolov5n_gray_silu_baseline/results.png
runs/yolov5n_gray_silu_baseline/confusion_matrix.png
runs/yolov5n_gray_silu_baseline_test/
```

回到工程根目录，计算最终权重哈希：

```powershell
Set-Location 'C:\Study\IC_Study\Project\FPGA\Contest\CGEDC_1\FPGA-YOLO'
Get-FileHash '.\runs\yolov5n_gray_silu_baseline\weights\best.pt' -Algorithm SHA256
```

在实验记录中填写训练命令、数据集版本、随机种子、每类 Precision/Recall、mAP50、mAP50-95、混淆矩阵和典型误检图。只有 SiLU 基线精度可接受，才开始训练同拓扑、同数据、同参数的 ReLU 版本；不要直接将这份 SiLU 权重部署到 FPGA。

## 7.绘图

```python
python tools\plot_yolov5_training.py `
  --run-dir runs\yolov5n_gray_silu_baseline2 `
  --data lv_yolo.yaml `
  --device 0
```



## 常见问题

- `ModuleNotFoundError`：确认已激活 `zynq-defect`，并在 `third_party\yolov5` 目录运行训练命令。
- `FileNotFoundError`：确认工程根目录是 `FPGA-YOLO`，且 `lv_yolo.yaml` 与 `data\LV-YOLO-ROI256` 未被移动。
- 显存不足：仅降低 `--batch-size`，例如从 `-1` 改为 `8` 或 `4`；保持 `--imgsz 256 --channels 1` 不变。
- 训练目录已存在：修改 `--name` 为新的实验名；不要在不确认内容的情况下使用 `--exist-ok` 覆盖旧结果。
