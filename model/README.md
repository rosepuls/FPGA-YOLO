# GC10-DET 数据集转换工具

脚本位置：

```text
model/convert_gc10_to_coco.py
```

原始数据默认位置：

```text
model/Data/GC10-DET
```

转换输出默认位置：

```text
model/Data/GC10-DET-COCO
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
python model/convert_gc10_to_coco.py
```

检查结果输出到终端，不创建转换数据。

## 3. 正式转换

```powershell
python model/convert_gc10_to_coco.py --convert
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
python model/convert_gc10_to_coco.py --convert --overwrite
```

## 4. 验证已有结果

```powershell
python model/convert_gc10_to_coco.py --validate model/Data/GC10-DET-COCO
```

返回结果中的 `valid` 必须是 `true`。

## 5. 常用调整

减少正常样本比例：

```powershell
python model/convert_gc10_to_coco.py --convert --negative-ratio 0.15
```

指定少数类目标实例数：

```powershell
python model/convert_gc10_to_coco.py --convert --minority-target 300
```

关闭增强以检查纯转换结果：

```powershell
python model/convert_gc10_to_coco.py --convert --no-negative-augmentation --no-minority-augmentation
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
