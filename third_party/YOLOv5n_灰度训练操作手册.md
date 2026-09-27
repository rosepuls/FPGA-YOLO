# YOLOv5n 单通道灰度模型训练操作手册

本文档适用于 Windows PowerShell，项目目录固定为：

```text
C:\Study\IC Study\Project\FPGA_Vivado\Contest\CGEDC_1\FPGA-YOLO
```

训练配置为：

- YOLOv5n v7.0
- 输入：`256×256×1` 灰度图
- 类别数：10
- 激活函数：SiLU
- 预训练权重：官方 `yolov5n.pt`
- 首层权重：RGB 三通道权重按通道均值转换为单通道
- 默认训练轮数：300
- 默认 batch size：32
- 默认设备：CUDA `0`
- 输出目录：`runs/yolov5n_gray256_silu_baseline/`

## 1. 打开 PowerShell

在 Windows 开始菜单中打开 **Windows PowerShell**，或者在项目文件夹空白处按住 `Shift` 后右键，选择“在此处打开 PowerShell”。

如果 PowerShell 尚未加载 Conda 命令，可以先执行：

```powershell
conda init powershell
```

执行后关闭当前 PowerShell，重新打开一个 PowerShell 窗口。

## 2. 进入项目目录

```powershell
Set-Location "C:\Study\IC Study\Project\FPGA_Vivado\Contest\CGEDC_1\FPGA-YOLO"
```

确认当前位置：

```powershell
Get-Location
```

输出路径应为：

```text
C:\Study\IC Study\Project\FPGA_Vivado\Contest\CGEDC_1\FPGA-YOLO
```

## 3. 激活训练环境

```powershell
conda activate zynq-defect
```

确认 Python 和 PyTorch：

```powershell
python --version
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not found')"
```

应当看到 `CUDA available: True`，并显示 NVIDIA GPU 名称。

## 4. 训练前预检

正式训练前先运行：

```powershell
python tools/train_yolov5n_silu.py --preflight-only
```

预检会检查：

- 数据集是否能读取为单通道张量；
- 输入 batch 是否为 `[2, 1, 256, 256]`；
- 模型首层是否为 1 个输入通道；
- 三个检测头是否均为 45 个输出通道；
- 官方 RGB 预训练权重是否已正确转换为灰度首层；
- CUDA 是否可用。

最后应看到类似输出：

```text
"batch_shape": [2, 1, 256, 256]
"first_conv_in_channels": 1
"head_out_channels": [45, 45, 45]
"pretrained_first_conv_verified": true
"status": "ready"
```

只有 `status` 为 `ready` 时才开始正式训练。

## 5. 开始正式训练

```powershell
python tools/train_yolov5n_silu.py
```

默认使用配置文件：

```text
configs/yolov5n_gray_silu_baseline.yaml
```

训练结果保存在：

```text
runs/yolov5n_gray256_silu_baseline/
```

重点文件包括：

```text
runs/yolov5n_gray256_silu_baseline/weights/best.pt
runs/yolov5n_gray256_silu_baseline/weights/last.pt
runs/yolov5n_gray256_silu_baseline/results.csv
runs/yolov5n_gray256_silu_baseline/results.png
```

训练过程中可以使用 `Ctrl+C` 安全中断。YOLOv5 会周期性保存 `last.pt`；中断前最好等待当前 epoch 保存完成。

## 6. 断点继续训练

如果训练被 `Ctrl+C`、关机或其他原因中断，从最近保存的权重继续：

```powershell
python third_party/yolov5/train.py --resume "runs/yolov5n_gray256_silu_baseline/weights/last.pt"
```

断点续训会读取该 run 保存的 `opt.yaml`，恢复原来的：

- epoch 进度；
- batch size；
- 学习率和优化器状态；
- 数据集配置；
- 单通道 `channels=1` 配置；
- 随机种子和其他训练选项。

如果实际权重路径不同，替换命令末尾的路径即可，例如：

```powershell
python third_party/yolov5/train.py --resume "runs/yolov5n_gray256_silu_baseline2/weights/last.pt"
```

如果目录中没有 `last.pt`，不能使用断点续训，需要检查 `weights` 目录是否存在，或者重新执行正式训练命令。

## 7. 查看训练过程

查看当前 run 的文件：

```powershell
Get-ChildItem "runs/yolov5n_gray256_silu_baseline"
Get-ChildItem "runs/yolov5n_gray256_silu_baseline\weights"
```

查看最新日志末尾内容：

```powershell
Get-Content "runs/yolov5n_gray256_silu_baseline\results.csv" -Tail 10
```

如果需要实时观察训练窗口，保持正在运行训练的 PowerShell 不要关闭。另开一个 PowerShell 窗口查看 GPU：

```powershell
nvidia-smi
```

## 8. 训练完成后评估验证集

训练完成后，使用最优权重评估验证集：

```powershell
python third_party/yolov5/val.py `
  --weights "runs/yolov5n_gray256_silu_baseline/weights/best.pt" `
  --data "gc10_yolo.yaml" `
  --imgsz 256 `
  --channels 1 `
  --task val
```

PowerShell 中每行末尾的反引号 `` ` `` 表示命令续行，也可以写成单行：

```powershell
python third_party/yolov5/val.py --weights "runs/yolov5n_gray256_silu_baseline/weights/best.pt" --data "gc10_yolo.yaml" --imgsz 256 --channels 1 --task val
```

## 9. 最终评估测试集

测试集只用于最终报告，不要在反复调参时频繁使用：

```powershell
python third_party/yolov5/val.py `
  --weights "runs/yolov5n_gray256_silu_baseline/weights/best.pt" `
  --data "gc10_yolo.yaml" `
  --imgsz 256 `
  --channels 1 `
  --task test
```

单行版本：

```powershell
python third_party/yolov5/val.py --weights "runs/yolov5n_gray256_silu_baseline/weights/best.pt" --data "gc10_yolo.yaml" --imgsz 256 --channels 1 --task test
```

## 10. 常见问题

### 10.1 `conda activate` 不可用

执行：

```powershell
conda init powershell
```

关闭并重新打开 PowerShell 后再次执行：

```powershell
conda activate zynq-defect
```

### 10.2 CUDA 不可用

确认当前激活的是 `zynq-defect` 环境：

```powershell
conda env list
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果显示 `False`，先不要开始训练，检查 NVIDIA 驱动和 PyTorch CUDA 安装。

### 10.3 找不到 `yolov5n.pt`

重新运行预检：

```powershell
python tools/train_yolov5n_silu.py --preflight-only
```

预检会自动下载官方 YOLOv5n 权重并放在项目根目录：

```text
C:\Study\IC Study\Project\FPGA_Vivado\Contest\CGEDC_1\FPGA-YOLO\yolov5n.pt
```

### 10.4 训练时提示输出目录已经存在

默认配置中的 `exist_ok` 为 `false`，用于避免覆盖已有实验。请确认旧实验是否需要保留；如需新实验，复制配置文件并修改 `name`，不要直接删除旧的 `runs` 目录。

### 10.5 训练后如何进入下一阶段

先保存以下结果：

- `best.pt` 和 `last.pt`；
- `results.csv`；
- val/test 的 Precision、Recall、mAP50、mAP50:95；
- 每类 AP 和 Recall；
- 混淆矩阵及预测可视化。

SiLU 基线完成后，才能继续 LeakyReLU 部署候选、INT8 量化和 FPGA Golden Model 对拍。
