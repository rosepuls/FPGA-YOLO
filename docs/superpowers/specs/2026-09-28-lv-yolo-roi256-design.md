# LV-COCO 到 YOLO ROI256 数据集转换设计

## 目标

从 `data/LV-COCO` 生成独立的 `data/LV-YOLO-ROI256` 训练集，供当前
YOLOv5n 单通道、`256 x 256` FPGA 部署链路使用。原始 LV-COCO 文件不得被
修改。

## 输入范围与完整性规则

- 只使用 `train2017` 与 `val2017` 中、且与其 COCO JSON 标注一一配对的图像。
- 不使用现有 `test2017`：其 JSON 的 165 条图片记录中只有 50 个同名文件，
  因而不能作为可信测试划分。
- 输入类别固定为 COCO ID `1..4`，按升序映射为 YOLO ID `0..3`：
  `ca_shang`、`zang_wu`、`zhe_zhou`、`zhen_kong`。
- 转换前验证图片元数据、COCO 引用关系和 bbox；任何缺图、非法类别或越界框
  都应终止转换，而不是静默跳过。

## 划分策略

- 合并可用的原训练、验证记录；不沿用它们原有的 split。
- 文件名去除末尾 `_aug_<整数>` 后得到 group key；同一个 group key 的原图和
  全部增强图必须进入同一 split。
- 以固定随机种子进行可复现的、类别覆盖优先的 group-level `80/10/10`
  train/val/test 划分。每一类至少在每个 split 出现一次，否则报错。
- 转换报告记录来源样本数、group 数、最终各 split 的图像数/目标数/类别数、
  随机种子和被排除的原测试集原因。

## 图像和标签转换

- 图像输入只接受 `RGB` 或 `L`。RGB 按项目数据契约的固定公式
  `(77R + 150G + 29B + 128) >> 8` 转为 `gray_u8`；`L` 原样保留。
- 将完整 `640 x 480` 画面作为临时 ROI，直接双线性缩放至 `256 x 256`；不补边。
  因各坐标轴独立缩放，COCO bbox 归一化后可直接写成 YOLO 标注。
- 输出图像为 `L` 模式 PNG；每张图都生成同名 `.txt`，可信负样本使用空标签。
- 输出必须经配对、尺寸、模式、类别范围与框几何验证。

## 交付物

- `models/convert_lv_coco_to_yolo_roi256.py`：可复用的转换器与 CLI。
- `models/tests/test_lv_yolo_roi256.py`：覆盖分组不泄漏、灰度转换、标签映射、
  输出验证与失败处理的测试。
- `data/LV-YOLO-ROI256/`：生成的图像、标签与转换报告。
- `lv_yolo.yaml`：YOLOv5 数据配置，指向新数据集并声明四类名称。
