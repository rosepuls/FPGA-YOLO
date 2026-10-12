# AC550 单次量化边界 QAT 设计

## 目标

消除当前 YOLOv5n-ReLU QAT 图对同一激活的重复 fake-quant，使训练期量化误差与 AC550 的单次 INT8 边存储一致，并以冻结 QAT 阶段的候选模型进行质量验收。

## 已观察到的问题

- `qat_v3` 在启用 QAT 后，验证集 mAP@0.5:0.95 从 FP32 基线的 `0.61957` 降至最高 `0.54357`。
- 当前 QAT 包装器会在生产者输出、消费者输入、C3 分支、Concat 和 Upsample 等连续位置再次 fake-quant 同一张量。
- 训练框架默认 `best.pt` 可能在 collect 阶段产生；它没有启用 fake-quant，不能作为部署候选。

## 量化边界契约

- 模型输入在第一层 Conv 前量化一次，使用 AC550 signed symmetric INT8，范围 `[-127, 127]`。
- 每个普通 Conv 和每个 Detect Conv 仅在输出处形成一个量化激活边；后继顺序 Conv 直接消费该边，不再独立量化输入。
- 普通 Conv 的权重在 Conv+BN 推理融合后的等效权重域中进行按输出通道 fake-quant；Detect 不使用 ReLU。
- Add 和 Concat 的输入已经来自已量化边。它们只在合并结果处量化一次为声明的输出尺度。
- MaxPool 与 nearest Upsample 保留输入尺度，训练模拟中不得再次 fake-quant。
- QAT 导出只接受 observer 已冻结、BN 统计已冻结的 checkpoint。训练器在 `max(freeze_observer_epoch, freeze_bn_epoch)` 之后独立保存 `weights/best_qat.pt`。

## 实施边界

本次只修改 QAT 图模拟、训练候选选择和相关单元测试。QAT 包导出器及 G2 验收程序在重新训练达到模型质量门槛后再改造。

## 验收标准

- 单元测试证明顺序 Conv、MaxPool、Upsample 不会对同一激活边重复调用 fake-quant。
- 单元测试证明 Add/Concat 只生成一个合并输出量化边。
- QAT collect 阶段的浮点输出保持与原 Conv+BN+ReLU 等价。
- 训练运行开始即生成 `qat_metadata.json`；epoch 20 及以后生成的 `best_qat.pt` 才可作为导出候选。
- 重训后在 validation split 上的 mAP@0.5 损失不超过 FP32 基线的 `1.0` 个百分点；未达标则不导出。
