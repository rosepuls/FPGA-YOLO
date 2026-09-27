# GC10-DET 数据与模型工具

脚本位置：

```text
models/convert_gc10_to_coco.py
```

原始数据默认位置：

```text
data/GC10-DET
```

转换输出默认位置：

```text
data/GC10-DET-COCO
```

脚本不会修改原始数据。未添加 `--convert` 时只执行检查。

## 1. 环境

激活项目环境并确认依赖：

```powershell
conda activate zynq-defect
python -m pip install pillow numpy
```

## 2. 只读检查

从仓库根目录运行：

```powershell
python models/convert_gc10_to_coco.py
```

检查结果输出到终端，不创建转换数据。

## 3. 正式转换

```powershell
python models/convert_gc10_to_coco.py --convert
```

默认行为包括：

- 合并 `10_yaozhe` 和 `10_yaozhed`；
- 隔离未知标签、无可靠标注和无法唯一匹配的图片；
- 将有效图片重新编码成单通道 PNG；
- 按材料/生产批次分组划分 70%/15%/15%；
- 只在训练集扩充少样本类别；
- 从远离缺陷框的区域裁取正常样本；
- 对少样本和正常样本使用不同的非几何增强策略；
- 生成 COCO 标注和审计报告。

如果输出目录已经存在，脚本会拒绝覆盖。明确需要重新生成时使用：

```powershell
python models/convert_gc10_to_coco.py --convert --overwrite
```

## 4. 验证已有结果

```powershell
python models/convert_gc10_to_coco.py --validate data/GC10-DET-COCO
```

返回结果中的 `valid` 必须是 `true`。

## 5. 常用调整

减少正常样本比例：

```powershell
python models/convert_gc10_to_coco.py --convert --negative-ratio 0.15
```

指定少数类目标实例数：

```powershell
python models/convert_gc10_to_coco.py --convert --minority-target 300
```

关闭增强以检查纯转换结果：

```powershell
python models/convert_gc10_to_coco.py --convert --no-negative-augmentation --no-minority-augmentation
```

修改随机种子后会改变分组分配、增强参数和正常区域采样。为了复现实验，应固定 `--seed`。

## 6. 增加真实正常图片

优先采集与最终相机、镜头、光源和曝光一致的无缺陷图片。将这些图片加入数字类别目录，并为每张图片提供包含正确 `folder`、`filename`、`size`、但不包含 `object` 的 XML。转换器会把它们作为可信空标注正常样本。

不要把来源不明或仅仅缺少 XML 的图片直接作为正常图片；其中可能含有未标注缺陷。

## 7. 输出报告

```text
reports/audit_report.json
reports/quarantine.json
reports/unused_images.json
reports/split_manifest.json
reports/augmentation_manifest.json
```

`quarantine.json` 记录被隔离的 XML；`unused_images.json` 记录没有被有效 XML 消费的物理图片；`augmentation_manifest.json` 可追溯每个增强样本的来源、裁剪范围和参数。

## 8. YOLOv5n 单通道 SiLU 基线

训练入口使用原生 `256×256×1` 灰度张量。官方 `yolov5n.pt` 的第一层 RGB 卷积会按输入通道求均值后转换为单通道，其余可匹配权重保持不变。

先激活环境并执行预检：

```powershell
conda activate zynq-defect
python tools/train_yolov5n_silu.py --preflight-only
```

只有输出包含以下内容时才开始训练：

```text
"batch_shape": [2, 1, 256, 256]
"first_conv_in_channels": 1
"head_out_channels": [45, 45, 45]
"pretrained_first_conv_verified": true
"status": "ready"
```

启动固定的 300 轮基线训练：

```powershell
python tools/train_yolov5n_silu.py
```

训练配置位于 `configs/yolov5n_gray_silu_baseline.yaml`，默认输出到 `runs/yolov5n_gray256_silu_baseline/`。

从最近一次中断处恢复：

```powershell
python third_party/yolov5/train.py --resume runs/yolov5n_gray256_silu_baseline/weights/last.pt
```

训练完成后分别评估验证集和测试集：

```powershell
python third_party/yolov5/val.py --weights runs/yolov5n_gray256_silu_baseline/weights/best.pt --data gc10_yolo.yaml --imgsz 256 --channels 1 --task val
python third_party/yolov5/val.py --weights runs/yolov5n_gray256_silu_baseline/weights/best.pt --data gc10_yolo.yaml --imgsz 256 --channels 1 --task test
```

当前仅完成训练前配置和真实数据预检；必须在 300 轮训练及 val/test 评估结束后，才能把阶段 1 标记为完成。
