# AC550 + OV5640 工业视觉缺陷检测一条龙实施计划

> **实施范围：** 本计划覆盖从 OV5640 上电与视频流采集、FPGA 图像预处理、工件到位/有效帧判断、YOLOv5n-ReLU 异构推理、缺陷判定，到 UART/GPIO/显示调试输出的完整工程闭环。本文只描述实施计划；不代表其中代码、模型或硬件工程已经存在。

## 1. 最终系统目标与固定分工

### 1.1 最终目标

在小梅哥 AC550-SoC 上实现如下闭环：

```text
OV5640
  │  SCCB 配置 + XCLK + DVP 像素流
  ▼
FPGA：采集、帧同步、像素格式检查、灰度化、ROI、缩放、工件到位判断
  ▼
HPS DDR3：原始/调试帧、256×256 模型输入、模型权重、特征图、检测头
  ▼
FPGA：INT8 Conv + ReLU + MaxPool
ARM：Add + Concat + Upsample + 图调度
  ▼
ARM：Decode + NMS + 缺陷判定 + 工件结果状态机
  ▼
UART 日志 / GPIO PASS-FAIL-NG-ERROR / 可选视频叠加显示
```

### 1.2 模型和推理分工

模型固定为 YOLOv5n（nano）四分类单通道 ReLU 版本：输入为 `[1,1,256,256]`，类别顺序固定为：

1. `ca_shang`
2. `zang_wu`
3. `zhe_zhou`
4. `zhen_kong`

原始 YOLOv5n 默认激活是 SiLU。为适配 AC550 资源，本工程使用 **ReLU 版 YOLOv5n 并重新训练**；现有 `yolov5n.pt` 只可作为迁移学习初始化权重，不能直接作为最终 ReLU 部署模型。

| 模块/算子 | 执行位置 | 说明 |
| --- | --- | --- |
| OV5640 配置、DVP 采集、RGB565 转灰度、ROI、Resize、工件到位判断 | FPGA | 实时流式处理，避免先整帧送 ARM 再预处理 |
| Conv2D INT8、Requant、ReLU、SPPF MaxPool、Detect 最后卷积 | FPGA | Detect 最后卷积为线性输出，**不加 ReLU** |
| C3 残差 Add、Concat、FPN/PAN Upsample、SPPF Concat、tensor 生命周期 | ARM/HPS | 均在共享 DDR 的 INT8 tensor 上执行 |
| 图调度、DMA/缓存管理、Decode、NMS、工件结论、通信 | ARM/HPS | ARM 是总控，FPGA 是算子加速器 |

### 1.3 版本 1 的可验收输出

- 从 OV5640 连续稳定采到实际视频帧。
- 每个有效工件帧生成可复现的 `256×256` 灰度输入。
- 工件未到位时不启动 YOLO，而是输出 `NOT_READY`。
- 工件到位后运行 YOLOv5n-ReLU，输出 4 类缺陷的类别、置信度、bbox 和 `PASS/FAIL`。
- 输入帧、模型输入、P3/P4/P5 检测头和最终结果可逐级追踪。
- 检测、相机、DMA、模型包或帧 ID 异常时输出 `ERROR`，绝不把异常作为 `PASS`。

---

## 2. 全局数据、坐标和状态契约

### 2.1 相机与帧格式契约

第一版采用 OV5640 的 DVP 并行输出，目标格式为 **RGB565、640×480**。该分辨率便于先完成稳定采集和 ROI 调试；若现场最终采用其他 OV5640 DVP 格式或分辨率，必须同时更新相机寄存器表、帧计数、ROI 表、Golden image 和验收数据，不能只改一端。

| 项目 | 固定约定 |
| --- | --- |
| 相机总线 | SCCB（与 I²C 时序兼容）：SIOC、SIOD |
| 控制信号 | XCLK、RESET、PWDN |
| 像素信号 | DVP `D[7:0]`、PCLK、HREF、VSYNC |
| 输出格式 | RGB565，两个 8-bit DVP 周期组成一个像素 |
| 采集尺寸 | 640×480；有效像素数 307,200 |
| 帧号 | `frame_id` 从 0 递增；VSYNC 边界产生新帧号 |
| 传输顺序 | 行优先：`y=0..479`，每行 `x=0..639` |

物理管脚、XCLK 来源/频率、IO 电平、上拉和 OV5640 模组的 RESET/PWDN 有效电平必须从 AC550 原理图和相机模块说明书填入 QSF；计划中不编造管脚号。

### 2.2 图像预处理契约

RGB565 先扩展到 8-bit RGB：

```text
R8 = (R5 << 3) | (R5 >> 2)
G8 = (G6 << 2) | (G6 >> 4)
B8 = (B5 << 3) | (B5 >> 2)
gray = (77 * R8 + 150 * G8 + 29 * B8 + 128) >> 8
```

ROI 参数由 ARM 通过 CSR 写入：`roi_x`、`roi_y`、`roi_width`、`roi_height`。ROI 必须完全位于 `0≤x<640, 0≤y<480` 的有效画面内；越界、宽高为零、缩放配置错误均视为 `PREPROCESS_ERROR`。

ROI 使用双线性插值缩放为 `256×256` 单通道 `uint8`，输出地址为 `model_input_addr`。训练集的图像转换、Python Golden Model、FPGA RTL 与 ARM 调试程序必须使用相同公式、边界取样和四舍五入规则；禁止训练使用 letterbox 而硬件使用直接 Resize。

### 2.3 工件状态和最终结果契约

每帧状态固定为：

```text
EMPTY → CAPTURING → PREPROCESSING → READY → INFERENCING → POSTPROCESSING → RESULT_READY → RELEASED
```

结果优先级固定为：

```text
ERROR > NOT_READY > FAIL > PASS
```

| 结果 | 条件 |
| --- | --- |
| `ERROR` | 相机配置失败、帧格式/尺寸异常、ROI 越界、DMA/FPGA task 错误、模型 SHA 不一致、frame_id 错配、watchdog 超时 |
| `NOT_READY` | 图像有效但工件到位检测未通过；不运行模型 |
| `FAIL` | 至少一个缺陷框在阈值和 NMS 后有效 |
| `PASS` | 工件到位、整条推理链正常、没有有效缺陷框 |

工件到位检测第一版不靠臆测阈值：在调试模式采集“空工位”和“正常工件”样本后，使用可配置的 ROI 灰度均值范围、前景像素比例和连续稳定帧数三个条件。默认判定逻辑为：

```text
present = (mean_gray in [mean_min, mean_max])
       && (foreground_ratio >= fg_ratio_min)
       && (stable_frame_count >= stable_frames_required)
```

其中 `mean_min`、`mean_max`、`fg_ratio_min` 和 `stable_frames_required` 全部由 ARM 配置并记录在工艺配置中；现场标定后才固定数值。若后续增加光电传感器，可将其作为额外 AND 条件，但不改变图像管线。

---

## 3. 第一阶段：AC550、OV5640 和视频流硬件打通

**阶段目标：** 不接模型，只证明 OV5640 能被正确配置，DVP 时序正确，连续视频帧能稳定进入 FPGA，并可从 HPS 观察验证。

### 3.1 建立 AC550 最小 Quartus SoC 工程

**计划文件：**

- Create: `fpga/quartus/ac550_soc.qpf`
- Create: `fpga/quartus/ac550_soc.qsf`
- Create: `fpga/constraints/ac550_soc.qsf`
- Create: `fpga/constraints/ac550_soc.sdc`
- Create: `fpga/platform_designer/ac550_soc_system.qsys`
- Create: `fpga/rtl/common/ac550_top.sv`
- Create: `fpga/rtl/common/reset_sync.sv`
- Create: `fpga/rtl/common/clock_reset_manager.sv`

- [ ] **步骤 1：整理硬件连接表。** 从 AC550 原理图/手册列出 50 MHz 时钟、JTAG、HPS DDR、HPS UART、OV5640 的 XCLK/RESET/PWDN/SIOC/SIOD/D[7:0]/PCLK/HREF/VSYNC、调试 LED 与可选显示输出。每个信号记录方向、IO 标准、上拉要求、有效电平和时钟域。
- [ ] **步骤 2：写复位和时钟失败测试。** 在仿真中验证异步复位释放后在各时钟域同步解除；PCLK 域、系统域和 CNN 域之间没有直接组合跨域路径。
- [ ] **步骤 3：创建最小工程。** QSF 固定 `FAMILY "Cyclone V"` 与 `DEVICE 5CSXFC6C6U23I7`；SDC 声明板载输入时钟、PLL 派生时钟与时钟不确定度。Platform Designer 加入 HPS、HPS-to-FPGA bridge、FPGA-to-HPS bridge、on-chip RAM、DMA、IRQ mapper、clock/reset controller。
- [ ] **步骤 4：编译最小 `.sof`。**

  ```powershell
  qsys-generate fpga/platform_designer/ac550_soc_system.qsys --synthesis=VERILOG
  quartus_sh --flow compile fpga/quartus/ac550_soc
  ```

  预期：目标器件正确；TimeQuest 无未约束时钟和负 slack。
- [ ] **步骤 5：板级验收。** 下载 `.sof`，LED 心跳、UART 启动信息、HPS DDR 循环读写和 CSR 读写均通过。未通过时只排查时钟、复位、管脚和 SoC，不开始相机调试。

### 3.2 OV5640 SCCB 初始化和相机自检

**计划文件：**

- Create: `fpga/rtl/camera/ov5640_sccb_master.sv`
- Create: `fpga/rtl/camera/ov5640_reg_rom.sv`
- Create: `fpga/rtl/camera/ov5640_init_controller.sv`
- Create: `fpga/sim/ov5640_sccb_tb.sv`
- Create: `ps_app/include/ov5640_status.h`
- Create: `ps_app/src/ov5640_status.c`

**初始化顺序：**

```text
上电稳定等待 → PWDN/RESET 置位 → XCLK 稳定 → RESET 释放 → SCCB 读芯片 ID
→ 写软件复位 → 写时钟/PLL → 写 RGB565 DVP 输出格式 → 写 640×480 窗口/缩放
→ 写极性/时序 → 关闭或启用测试条 → 再次读关键寄存器 → 进入 STREAMING
```

- [ ] **步骤 1：写 SCCB 失败测试。** 覆盖 START、7-bit slave address、16-bit register address、8-bit data、ACK、STOP、NACK、重试次数和总线超时；NACK 不得进入 STREAMING。
- [ ] **步骤 2：实现可参数化 SCCB master。** 产生符合目标 SCCB 速率的 SIOC；SIOD 为开漏行为。寄存器表使用 ROM/数组，禁止把几十个寄存器写散在状态机 case 分支中。
- [ ] **步骤 3：实现相机初始化状态机。** 状态至少包括 `POWER_DOWN`、`RESET_ASSERT`、`XCLK_WAIT`、`READ_ID`、`WRITE_TABLE`、`VERIFY`、`STREAMING`、`ERROR`。错误寄存器记录失败步骤、寄存器地址和 NACK/timeout 类型。
- [ ] **步骤 4：首先开启 OV5640 彩条测试模式。** 在彩条模式下验证相机链路，不把第一次黑屏归因于 CNN。HPS UART 输出读取的 chip ID、初始化版本、状态、失败原因。
- [ ] **步骤 5：相机验收。** 必须连续完成 100 次初始化/复位循环且都读到正确 ID；彩条模式稳定输出前，不进入真实镜头和图像预处理阶段。

### 3.3 DVP 采集、帧同步和 RGB565 组像素

**计划文件：**

- Create: `fpga/rtl/camera/dvp_rx.sv`
- Create: `fpga/rtl/camera/dvp_pixel_unpack.sv`
- Create: `fpga/rtl/camera/dvp_async_fifo.sv`
- Create: `fpga/rtl/camera/frame_monitor.sv`
- Create: `fpga/sim/dvp_rx_tb.sv`
- Create: `fpga/sim/frame_monitor_tb.sv`

**DVP 输入状态：**

```text
WAIT_VSYNC → WAIT_HREF → BYTE0 → BYTE1 → PIXEL_VALID → LINE_END → FRAME_END
```

- [ ] **步骤 1：写 DVP 失败测试。** 仿真产生 VSYNC/HREF/PCLK 和 RGB565 字节流，验证两字节只组成一个像素；验证 HREF 结束时不能多取/少取像素；验证 VSYNC 中断行时丢弃不完整行并报告错误。
- [ ] **步骤 2：实现 PCLK 域采集。** 仅在 HREF 有效的 PCLK 边沿采样 `D[7:0]`；根据已验证的 OV5640 字节序组合 RGB565。若检测到行像素数不是 640 或帧行数不是 480，置 `FRAME_FORMAT_ERROR`。
- [ ] **步骤 3：实现跨时钟 FIFO。** DVP PCLK 域向系统时钟域传输 `{frame_id, x, y, rgb565, pixel_valid, line_last, frame_last}`；FIFO 溢出/欠读必须记录并使该帧变为 ERROR。
- [ ] **步骤 4：实现 frame monitor。** 统计帧率、每行像素、每帧行数、丢帧和 VSYNC 周期；通过 CSR/HPS UART 输出。调试期可将抽样像素或一帧 RGB565 写入 DDR 供软件查看。
- [ ] **步骤 5：彩条验收。** 对彩条帧验证红、绿、蓝、白、黑区域的 RGB565 取值及位置；连续采集 10 分钟，无 FIFO overflow、尺寸错误或帧号跳变。

---

## 4. 第二阶段：FPGA 图像预处理与工件到位判断

**阶段目标：** 从真实 OV5640 视频中稳定得到与模型训练一致的单通道 256×256 输入，并在模型前可靠地区分“无工件/工件未稳定/可推理”。

### 4.1 RGB565 灰度化、ROI 和双线性缩放

**计划文件：**

- Create: `fpga/rtl/preprocess/rgb565_to_gray.sv`
- Create: `fpga/rtl/preprocess/roi_controller.sv`
- Create: `fpga/rtl/preprocess/resize_bilinear_256.sv`
- Create: `fpga/rtl/preprocess/preprocess_dma_writer.sv`
- Create: `fpga/sim/rgb565_to_gray_tb.sv`
- Create: `fpga/sim/resize_bilinear_256_tb.sv`
- Create: `models/export/make_preprocess_golden.py`
- Create: `models/tests/test_preprocess_contract.py`

- [ ] **步骤 1：产生 Python Golden 输入。** 使用固定 RGB565 彩条、边界像素图、实际工件帧和不同 ROI 生成灰度、ROI、256×256 输出；每个样本保存输入、ROI 参数和 `golden_input_u8.bin`。
- [ ] **步骤 2：写灰度失败测试。** 覆盖 RGB565 的黑/白/纯红/纯绿/纯蓝、R5/G6/B5 最大值和随机像素；输出必须等于本计划第 2.2 节公式。
- [ ] **步骤 3：写 Resize 失败测试。** 覆盖 ROI 等于整图、最小合法 ROI、边界 ROI、非整数缩放比和四角像素；每个输出字节与 Python Golden 相等。
- [ ] **步骤 4：实现流式预处理。** `rgb565_to_gray` 在系统时钟域输出灰度流；`roi_controller` 丢弃 ROI 外像素；`resize_bilinear_256` 使用 line buffer 与固定点坐标计算。输出必须严格为 256 行、每行 256 字节。
- [ ] **步骤 5：实现 DDR 写入和完成事件。** 每帧 `model_input_addr` 64-byte 对齐；写入完成生成 `PREPROCESS_DONE(frame_id, input_crc)`。HPS 在收到事件前不允许启动第一层推理。
- [ ] **步骤 6：预处理验收。** FPGA 输出与 Golden 输入逐字节一致；对连续 1000 帧检查没有串帧、行错位、ROI 越界漏报或输出长度错误。

### 4.2 工件到位、稳定帧与触发策略

**计划文件：**

- Create: `fpga/rtl/preprocess/piece_presence.sv`
- Create: `fpga/rtl/preprocess/frame_stability.sv`
- Create: `fpga/sim/piece_presence_tb.sv`
- Create: `ps_app/include/inspection_config.h`
- Create: `ps_app/src/inspection_config.c`
- Create: `ps_app/tests/test_piece_presence_config.c`

**第一版检测指标：**

| 指标 | 计算位置 | 含义 |
| --- | --- | --- |
| `mean_gray` | FPGA | ROI 内像素灰度平均值 |
| `foreground_ratio` | FPGA | 满足 `[fg_low, fg_high]` 的 ROI 像素比例 |
| `frame_delta` | FPGA | 当前帧与上一帧抽样像素的绝对差平均值 |
| `stable_frame_count` | FPGA | 连续满足 `frame_delta <= delta_max` 的帧数 |
| `present` | FPGA/ARM | 灰度范围、前景比例和稳定帧数同时通过 |

- [ ] **步骤 1：采集工艺标定样本。** 分别保存空工位、正常工件、工件刚进入、工件移动、强/弱光照样本；记录各指标的统计分布。此步骤是为阈值配置提供数据，不能凭视觉估计写阈值。
- [ ] **步骤 2：写失败测试。** 空工位应为 `NOT_READY`，正常静止工件应在规定稳定帧数后变为 `READY`，移动工件应回到 `NOT_READY`；配置范围非法时报告参数错误。
- [ ] **步骤 3：实现统计与稳定状态机。** `EMPTY`、`CANDIDATE`、`STABILIZING`、`READY`、`LOST` 五态；任何帧格式错误、ROI 错误或 FIFO 错误直接进入 `ERROR`，不允许误认为工件离开。
- [ ] **步骤 4：实现 HPS 参数配置。** HPS 可通过配置文件/CSR 设置阈值；每次推理日志必须带上配置版本和实际统计值，便于追溯误检原因。
- [ ] **步骤 5：工件到位验收。** 在标定样本上统计 false-ready、false-not-ready 和稳定等待帧数；只有工件状态为 READY 且输入 CRC 完整时，才转入模型推理。

### 4.3 调试画面、帧缓存和视频观察

**计划文件：**

- Create: `fpga/rtl/video/debug_overlay.sv`
- Create: `fpga/rtl/video/framebuffer_writer.sv`
- Create: `ps_app/src/frame_dump.c`
- Create: `ps_app/tests/test_frame_dump.c`

- [ ] **步骤 1：建立调试帧格式。** 原始 RGB565、灰度 ROI、256×256 模型输入和最终检测框使用不同 DDR buffer/文件名，不能覆盖模型输入。
- [ ] **步骤 2：实现可选叠加。** 至少可显示/导出 frame_id、ROI 矩形、`NOT_READY/READY/ERROR`、mean_gray、foreground_ratio；若板卡显示接口尚未确定，第一版用 HPS 导出帧文件或串口 CRC 替代，不阻塞主链路。
- [ ] **步骤 3：实现帧导出。** HPS 可按 frame_id 保存原始帧、ROI、模型输入和后续结果，供 Python Golden Model 回放。
- [ ] **步骤 4：调试验收。** 同一 frame_id 的原始帧、ROI、256 输入和状态统计能对应，不能出现“画面来自 A 帧、推理来自 B 帧”。

---

## 5. 第三阶段：训练 ReLU YOLOv5n、量化并导出异构模型包

**阶段目标：** 获得与硬件拆分完全一致的四分类 ReLU INT8 模型包。该阶段不要求 FPGA 整网运行，但必须能在 Python 中按“FPGA 节点/ARM 节点”执行并产生 Golden tensors。

### 5.1 ReLU 单通道四分类模型训练

#### 前置基线：SiLU 模型训练与初步验证（已完成）

在开始 ReLU 重训前，已用 YOLOv5n 默认 SiLU 激活完成单通道、四分类、`256×256` 的精度基线训练。训练使用 LV-YOLO-ROI256 数据集、`yolov5n.pt` 迁移学习初始化、100 epochs、batch size 32、随机种子 `20260930`。该基线用于检查数据和任务可学性，并作为 ReLU 精度比较对象；SiLU 权重不符合 FPGA 的 ReLU 部署契约。

使用该次训练的 `best.pt` 在独立 test split 上验证，测试集共 256 张图、457 个缺陷实例：

| 类别 | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| `ca_shang` | 0.983 | 0.974 | 0.985 | 0.696 |
| `zang_wu` | 1.000 | 0.950 | 0.978 | 0.667 |
| `zhe_zhou` | 0.838 | 0.841 | 0.755 | 0.464 |
| `zhen_kong` | 0.962 | 0.966 | 0.970 | 0.499 |
| **总体** | **0.946** | **0.933** | **0.922** | **0.582** |

产物位于 `runs/yolov5n_gray_silu_baseline2/`：权重为 `weights/best.pt`，训练曲线数据为 `results.csv`；测试集混淆矩阵及 P/R/F1/PR 曲线位于 `analysis/validation/`。测试集结果表明 SiLU 基线初步可用，其中 `zhe_zhou` 是当前主要短板，进入部署前需优先检查该类样本和标注，并在 ReLU 重训后按相同数据划分复测。

训练的 100 个 epoch 已完成并产出可加载的 `best.pt`；训练结束的 optimizer 清理步骤曾因 PyTorch 2.11 默认 `weights_only=True` 报错。该收尾兼容问题不影响已保存权重和本节单独完成的测试集验证，但应修正 `strip_optimizer()` 的 checkpoint 加载调用，避免后续训练结束时再次报错。

**计划文件：**

- Create: `configs/yolov5n_gray_relu_256.yaml`
- Create: `tools/train_yolov5n_gray_relu_256.py`
- Create: `models/tests/test_yolov5n_relu_contract.py`
- Modify: `models/tests/test_yolov5_gray_model.py`
- Modify: `models/tests/test_yolov5_gray_dataloader.py`

- [ ] **步骤 1：写模型契约失败测试。** 断言 `ch=1`、`nc=4`、普通 Conv 为 ReLU、三个 Detect 卷积均为 27 通道线性输出；禁止出现 SiLU/80 类/10 类/45 通道残留。
- [ ] **步骤 2：实现 ReLU 配置。** 保持 YOLOv5n 的深度/宽度、C3、SPPF、FPN、PAN 和 P3/P4/P5 拓扑，只替换激活函数为 ReLU；不能为省资源删除网络分支而不重做训练/接口。
- [ ] **步骤 3：执行迁移学习。**

  ```powershell
  Set-Location third_party/yolov5
  python train.py --weights ../../yolov5n.pt --cfg ../../configs/yolov5n_gray_relu_256.yaml --data ../../lv_yolo.yaml --imgsz 256 --channels 1 --epochs 100 --batch-size -1 --project ../../runs --name yolov5n_gray_relu_256
  ```

- [ ] **步骤 4：验证与冻结。** 使用 `val.py` 对 `best.pt` 验证并记录每类 Precision、Recall、mAP50、mAP50-95、混淆矩阵；记录模型 SHA-256、训练命令、数据版本和 YOLOv5 commit。
- [ ] **步骤 5：G1 验收。** `best.pt` 必须实际为单通道四分类 ReLU 网络，且与预处理 Golden 的输入格式一致；未达到现场可接受精度时先调整训练，不进入量化或 RTL。

### 5.2 INT8 量化与模型图切分

**计划文件：**

- Create: `models/export/quantize_yolov5n_relu_ac550.py`
- Create: `models/export/export_yolov5n_relu_ac550.py`
- Create: `models/tests/test_yolov5n_relu_export.py`
- Create: `models/tests/vectors/`（由导出器生成）
- Modify: `docs/plans/yolov5图像接口约束.txt`

**导出模型包：** `runs/ac550_yolov5n_relu_int8/manifest.json`、`graph.json`、`weights_i8.bin`、`bias_i32.bin`、`quant_params.json`、`anchors.json`、`golden_input_u8.bin`、每个图节点的 Golden tensor、`head_p3_i8.bin`、`head_p4_i8.bin`、`head_p5_i8.bin`。

- [ ] **步骤 1：写导出失败测试。** 每个 graph node 必须带 `executor`：`FPGA_CONV_RELU`、`FPGA_CONV_LINEAR`、`FPGA_MAXPOOL`、`ARM_ADD`、`ARM_CONCAT`、`ARM_UPSAMPLE` 六者之一。Detect 最后一层只允许 `FPGA_CONV_LINEAR`。
- [ ] **步骤 2：融合与量化。** 融合 Conv+BN；权重/激活为 int8，bias/累加为 int32；每层导出 scale、zero point、integer multiplier、right shift、rounding、saturation。若 PTQ 误差超标，采用 QAT 重训，但不更改 runtime 文件格式。
- [ ] **步骤 3：写图分割规则。** C3 内的 Conv 下发 FPGA，残差 Add/Concat 交 ARM；SPPF 的 3 次 MaxPool 下发 FPGA、4 路 Concat 交 ARM；FPN/PAN 的 Upsample/Concat 交 ARM；每个检测头的最终 Conv 下发 FPGA、Decode/NMS 交 ARM。
- [ ] **步骤 4：导出跨边界 Golden。** 保存每个 FPGA 任务输入/输出、每个 ARM 算子输入/输出、三路 head 和最终检测框。任何不同 manifest SHA 的 Golden 均不得混用。
- [ ] **步骤 5：G2 验收。** Python INT8 执行结果与 FP32 ReLU 模型在相同输入/阈值下对比；记录逐层误差、框坐标、类别和置信度差。量化精度达标后冻结 manifest SHA。

---

## 6. 第四阶段：FPGA Conv + ReLU + MaxPool 加速器

**阶段目标：** FPGA 成为由 ARM 描述符驱动的 INT8 算子执行器，不负责解释整个 YOLOv5 图；所有结果可以独立与第三阶段 Golden tensor 比对。

### 6.1 CSR、DMA、任务接口

**计划文件：**

- Create: `fpga/rtl/avalon/ac550_cnn_csr.sv`
- Create: `fpga/rtl/avalon/cnn_job_controller.sv`
- Create: `fpga/rtl/cnn/tensor_dma.sv`
- Create: `fpga/rtl/cnn/fpga_op_executor.sv`
- Create: `fpga/sim/ac550_cnn_csr_tb.sv`
- Create: `fpga/sim/fpga_op_executor_tb.sv`

**任务描述符：**

| 字段 | 用途 |
| --- | --- |
| `opcode` | `CONV_RELU`、`CONV_LINEAR`、`MAXPOOL` |
| `task_id` / `frame_id` | 防止中断和结果串帧 |
| `input_addr` / `weight_addr` / `bias_addr` / `output_addr` | HPS DDR3 物理地址，64-byte 对齐 |
| `input_shape` / `output_shape` | NCHW 维度 |
| `kernel_stride_pad` | Conv/Pool 参数 |
| `quant_id` | manifest 内的量化条目 |
| `status` / `error_code` / `done_task_id` | 运行、错误和完成回报 |

- [ ] **步骤 1：写 CSR 失败测试。** 覆盖描述符锁存、busy 时拒绝 START、地址未对齐、未知 opcode、watchdog、完成 IRQ 和 `done_task_id`。
- [ ] **步骤 2：实现 DMA。** 以 Avalon burst 从 DDR 读 input/weight tile，向 DDR 写 output tile；读写计数必须与 descriptor shape 匹配，错误不允许继续写下一帧。
- [ ] **步骤 3：实现任务状态机。** `IDLE → LOAD → EXECUTE → STORE → DONE` 或 `ERROR`。全部 output 写回成功后才更新完成 ID 和 IRQ。
- [ ] **步骤 4：任务级仿真。** 连续运行 `CONV_RELU → MAXPOOL → CONV_LINEAR`，检查任务顺序、DDR 字节数、task ID、错误码和中断顺序。
- [ ] **步骤 5：SoC 验收。** HPS 可提交一个测试 Conv 任务并收回 Golden 输出；任何 cache 未清、地址不对齐或超时都应明确报错。

### 6.2 INT8 Conv、Requant 和 ReLU

**计划文件：**

- Create: `fpga/rtl/cnn/conv2d_i8.sv`
- Create: `fpga/rtl/cnn/requant_i32_to_i8.sv`
- Create: `fpga/rtl/cnn/relu_i8.sv`
- Create: `fpga/rtl/cnn/feature_tile_buffer.sv`
- Create: `fpga/rtl/cnn/weight_tile_buffer.sv`
- Create: `fpga/sim/conv2d_i8_tb.sv`
- Create: `fpga/sim/relu_i8_tb.sv`

- [ ] **步骤 1：写卷积失败测试。** 覆盖首层 `6×6 stride 2`、普通 `3×3`、`1×1`、padding、stride、负权重、负激活、int32 累加边界和 int8 饱和，逐元素比较 Golden。
- [ ] **步骤 2：写 ReLU 失败测试。** `[-128,-1,0,1,127]` 必须变为 `[0,0,0,1,127]`；`CONV_LINEAR` 必须保留负数，保证 Detect logits 正确。
- [ ] **步骤 3：实现 MAC 阵列。** INT8×INT8 乘法、INT32 累加、bias、Requant、ReLU 按流水执行。`Pin/Pout` 和 tile 尺寸参数化，由 Fitter/TimeQuest 结果而非主观猜测确定。
- [ ] **步骤 4：验证每层。** 用第三阶段导出的每一个 Conv 边界 tensor 测试 FPGA；C3 中的不同分支、P3/P4/P5 head 均需覆盖。
- [ ] **步骤 5：资源/时序验收。** 记录 DSP、LE、M10K、100 MHz worst slack、每层延迟和 DDR 带宽；若资源不足，只降低并行度/tile 并重新回归。

### 6.3 SPPF MaxPool

**计划文件：**

- Create: `fpga/rtl/cnn/maxpool_i8.sv`
- Create: `fpga/sim/maxpool_i8_tb.sv`

- [ ] **步骤 1：写失败测试。** 覆盖 `5×5,stride=1,pad=2` 的中心、边缘、角点，输入包含全部负数，验证 MaxPool 不隐含 ReLU。
- [ ] **步骤 2：实现固定配置 line buffer。** 第一版只支持 SPPF 所需参数，其他参数返回 `ERROR_UNSUPPORTED_OP`。
- [ ] **步骤 3：与软件 Golden 对比。** 三次串行 MaxPool 的每一次输出均比对，不只比最终 SPPF 输出。
- [ ] **步骤 4：接入统一 executor。** MaxPool 与 Conv 共用 task/DDR/IRQ 接口，ARM 通过 graph node 选择 opcode。
- [ ] **步骤 5：完成 G3。** 在实际 AC550 配置下同时满足资源、时序和单算子 Golden 回归，才允许接入 ARM 图解释器。

---

## 7. 第五阶段：ARM 图调度、非卷积算子和检测后处理

**阶段目标：** ARM 将 YOLOv5n-ReLU 的完整拓扑串起来，并正确使用 FPGA 产生的 tensor/head，而不是只做 NMS。

### 7.1 模型包、共享 DDR 与帧/tensor 生命周期

**计划文件：**

- Create: `ps_app/include/ac550_cnn_regs.h`
- Create: `ps_app/include/model_manifest.h`
- Create: `ps_app/include/frame_pool.h`
- Create: `ps_app/include/tensor_pool.h`
- Create: `ps_app/src/ac550_cnn_regs.c`
- Create: `ps_app/src/model_manifest.c`
- Create: `ps_app/src/frame_pool.c`
- Create: `ps_app/src/tensor_pool.c`
- Create: `ps_app/tests/test_frame_pool.c`
- Create: `ps_app/tests/test_tensor_pool.c`

- [ ] **步骤 1：定义 DDR 分区。** `camera_debug`、`model_input`、`weights_ro`、`bias_ro`、`workspace`、`head_p3/p4/p5`、`descriptor` 分区独立，全部 64-byte 对齐；记录物理地址、长度、所有者、cache 操作和 frame ID。
- [ ] **步骤 2：实现模型 manifest 校验。** 校验 schema、模型 SHA、输入/类别、量化版本、CSR 版本和 graph executor 类型。校验失败直接 `ERROR`。
- [ ] **步骤 3：实现 frame pool。** `FREE → CAPTURING → PREPROCESSING → READY → INFERENCING → POSTPROCESSING → RESULT_READY → FREE`。同一 frame_id 的输入、workspace/head 和结果必须绑定。
- [ ] **步骤 4：实现 tensor 引用计数。** graph node 输出被全部消费者使用前不得释放；ARM 与 FPGA 共享 tensor 时完成 cache clean/invalidate。
- [ ] **步骤 5：运行生命周期测试。** 覆盖连续帧、busy、错误中止、重复中断、ID 错配；任何 `INFERENCING` buffer 被覆盖均视为严重失败。

### 7.2 ARM Add、Concat、Upsample 与 graph executor

**计划文件：**

- Create: `ps_app/include/yolov5_graph.h`
- Create: `ps_app/src/yolov5_graph.c`
- Create: `ps_app/src/arm_add_i8.c`
- Create: `ps_app/src/arm_concat_i8.c`
- Create: `ps_app/src/arm_upsample_i8.c`
- Create: `ps_app/tests/test_arm_ops.c`
- Create: `ps_app/tests/test_yolov5_graph.c`

- [ ] **步骤 1：写 ARM 算子失败测试。** Add 覆盖不同量化 scale、正负饱和；Concat 验证 NCHW C 维顺序；Upsample 验证 nearest `×2` 复制位置；全部与第三阶段 Golden tensor 比对。
- [ ] **步骤 2：实现 Add。** 输入 scale 不一致时先重标定到输出 scale，使用 int32 加法后饱和 int8；禁止直接把不同 scale 的二进制值相加。
- [ ] **步骤 3：实现 Concat/Upsample。** NCHW 连续 row-major 内存布局；shape/scale 不合约时立即报图错误，不能静默截断。
- [ ] **步骤 4：实现 graph executor。** 读取 `graph.json`，节点为 FPGA 类型则提交 descriptor 等待 IRQ；节点为 ARM 类型则执行本地算子。C3、SPPF、FPN、PAN 均按第三阶段切分规则执行，源码中不硬编码 YOLO 层号。
- [ ] **步骤 5：G4 整图 tensor 验收。** 用一个 Golden 输入从首层跑至 P3/P4/P5；每个节点记录 CRC/耗时，与 Python INT8 图逐节点对比，定位标签必须是 `FPGA_OP`、`ARM_OP` 或 `DDR_CACHE`。

### 7.3 P3/P4/P5 Decode、NMS 与工件结论

**计划文件：**

- Create: `ps_app/include/yolo_decode.h`
- Create: `ps_app/include/result_mapper.h`
- Create: `ps_app/src/yolo_decode.c`
- Create: `ps_app/src/result_mapper.c`
- Create: `ps_app/tests/test_yolo_decode.c`
- Create: `ps_app/tests/test_result_mapper.c`

- [ ] **步骤 1：写 head 布局失败测试。** NCHW 读取固定为 `head[(anchor*9+element)*H*W + y*W + x]`；P3/P4/P5 的 H/W 分别为 `32/16/8`。
- [ ] **步骤 2：实现反量化、sigmoid 和 anchor decode。** anchor、stride、阈值和量化参数来自 manifest；不得使用 COCO 的 80 类信息或硬编码默认 anchor。
- [ ] **步骤 3：实现 class-aware NMS。** NMS 用源图 ROI 坐标；同类按 IoU 阈值抑制，不同类独立处理。
- [ ] **步骤 4：实现工件结论器。** `NOT_READY` 不推理；任一有效缺陷框为 FAIL；无有效框且全链正确为 PASS；任一预处理/图执行/DMA/head/frame ID 问题为 ERROR。
- [ ] **步骤 5：Python/C 交叉验收。** 对相同三路 INT8 head，Python/C 的类别、框、置信度、NMS 和 PASS/FAIL 一致或在 manifest 写明的数值容差内。

---

## 8. 第六阶段：结果输出、现场调试和整机验收

**阶段目标：** 输出可接入产线的明确结果，并保留定位问题所需的图像、模型和时序证据。

### 8.1 UART、GPIO 与调试输出

**计划文件：**

- Create: `ps_app/include/result_output.h`
- Create: `ps_app/src/result_output_uart.c`
- Create: `ps_app/src/result_output_gpio.c`
- Create: `ps_app/tests/test_result_output.c`
- Create: `docs/plans/ac550-result-protocol.md`

**UART 文本协议（第一版）：**

```text
RESULT frame=<u32> state=<PASS|FAIL|NOT_READY|ERROR> class=<id|none>
score=<0.000..1.000> box=<x1,y1,x2,y2|none> latency_ms=<u32> error=<code|none>
```

**GPIO 约定（第一版）：** `PASS=1`、`FAIL=1`、`NOT_READY=1`、`ERROR=1` 四个独立状态信号中任意时刻最多一个有效；结果保持时间和复位方式由 ARM 配置，不能用短脉冲丢失工件结果。

- [ ] **步骤 1：写结果优先级失败测试。** 覆盖 ERROR 覆盖 PASS/FAIL、NOT_READY 不产生 PASS、同一 frame 只能输出一次终态、重复 IRQ 不重复输出。
- [ ] **步骤 2：实现 UART 结构化日志。** 每行输出 frame ID、模型 SHA 短码、工件统计、执行耗时、最终状态和错误码；日志可被 PC 脚本解析。
- [ ] **步骤 3：实现 GPIO。** 只在 `RESULT_READY` 后更新，ERROR 优先；清楚定义下一个工件到来时的清除条件。
- [ ] **步骤 4：实现调试工件包。** 对 FAIL/ERROR 帧保存原始帧、ROI/模型输入、关键 tensor CRC、head CRC、bbox、阈值和软件版本；PASS 帧按可配置采样率保存，避免占满存储。
- [ ] **步骤 5：输出验收。** 串口、GPIO、调试帧和内部 frame_id 逐一对应；断开相机、制造 DMA 错误和加载错误模型包时均输出 ERROR。

### 8.2 一条龙端到端验收

**计划文件：**

- Create: `ps_app/src/inspection_service.c`
- Create: `ps_app/tests/test_end_to_end_contract.c`
- Create: `docs/plans/ac550-ov5640-yolov5n-relu-验收记录.md`
- Modify: `fpga/README.md`
- Modify: `ps_app/README.md`

| 验收层级 | 输入 | 必须验证的结果 |
| --- | --- | --- |
| 相机 | OV5640 彩条/实拍 | SCCB、分辨率、字节序、帧率、frame_id |
| 预处理 | RGB565 实帧 + ROI | 灰度、ROI、Resize、输入 CRC、工件状态 |
| FPGA 算子 | Golden tensor | Conv/ReLU/MaxPool 按节点逐元素一致 |
| ARM 图 | Golden input | Add/Concat/Upsample、P3/P4/P5 CRC 一致 |
| 后处理 | 三路 head | Decode/NMS/类别/bbox/结论一致 |
| 整机 | 空工位、正常件、四类缺陷、移动件、异常件 | NOT_READY/PASS/FAIL/ERROR 正确且只输出一次 |

- [ ] **步骤 1：准备验收样本。** 至少包含空工位、正常工件、4 类缺陷各一个、工件移动、过曝/欠曝、ROI 越界配置、相机断连和故意 DMA 错误样本。
- [ ] **步骤 2：执行单帧全链路。** 严格按“OV5640 → DVP → 预处理 → 工件状态 → 图调度 → FPGA/ARM 算子 → Decode/NMS → UART/GPIO”比对每一个边界，不能只看最终 PASS/FAIL。
- [ ] **步骤 3：执行连续压力测试。** 连续运行至少 1 小时，统计采集帧数、READY 帧数、推理帧数、PASS/FAIL/NOT_READY/ERROR、丢帧、FIFO overflow、DMA timeout、frame ID 错配、最大延迟和平均延迟。
- [ ] **步骤 4：性能拆分。** 分别记录相机帧周期、预处理、FPGA Conv/ReLU/MaxPool、ARM Add/Concat/Upsample、Decode/NMS、结果输出和端到端时间；不只报一个“FPS”。
- [ ] **步骤 5：最终通过判定。** 同时满足：OV5640 视频稳定；预处理输入与 Golden 一致；工件状态判定正确；FPGA 实际执行全部 Conv/ReLU/MaxPool；ARM 实际执行其余图节点；四类缺陷结果正确；ERROR 从不误报 PASS；压力测试无 buffer/frame 串扰；Quartus TimeQuest 无未约束时钟和负 slack。

---

## 9. 推荐执行顺序与禁止跳过项

1. **先做相机彩条和 DVP 采集，再做真实工件图像。** 不允许在视频流尚不稳定时调 YOLO。
2. **先验证 RGB565 → 灰度 → ROI → 256 输入，再训练/量化。** 模型训练和板端输入不一致会使后续调试失去意义。
3. **先完成 ReLU 模型训练与 INT8 Golden，再写 FPGA 卷积。** ReLU 和 SiLU 权重/输出不能混用。
4. **先验证 FPGA 单算子，再接 ARM 图调度。** Conv/ReLU/MaxPool 必须逐层对比 Golden。
5. **先完成整图软件/软硬协同，再接输出接口。** UART/GPIO 只能消费已经确认正确的终态。
6. **任何 ERROR、frame_id 不一致、CRC 不一致、DMA timeout、相机格式异常均停止该帧。** 不允许为保持帧率而输出猜测结果。

## 10. 关键风险与对应策略

| 风险 | 处理方式 |
| --- | --- |
| OV5640 无画面/颜色错乱 | 固定先开彩条；验证 SCCB ID、RGB565 字节序、VSYNC/HREF/PCLK 和行列统计 |
| 训练输入与硬件输入不同 | 以 `make_preprocess_golden.py` 为单一参考，训练、RTL、HPS 都用同一向量回归 |
| ReLU 精度下降 | 先完成 ReLU 重训和量化评估；不在部署时将 ReLU 临时换回 SiLU |
| FPGA 资源不足 | 降低 `Pin/Pout` 或 tile 尺寸，权重放 DDR；不扩大模型、不强行全权重驻片上 |
| ARM/FPGA DDR 往返过多 | 记录每个节点 DMA 与 ARM 时间，按实测调整连续 tile group；不在无数据时改变分工 |
| 工件运动造成误检 | 用稳定帧状态机门控推理，并记录图像统计；必要时接入外部触发但不破坏 frame ID 契约 |
| 现场异常被当作良品 | `ERROR > NOT_READY > FAIL > PASS` 的结果优先级写入测试和输出模块 |

## 11. 最终交付清单

- AC550 Quartus/Platform Designer 工程、QSF/SDC、`.sof` 和资源/时序报告。
- OV5640 SCCB 寄存器表、初始化版本、相机/视频流自检记录。
- Python 预处理 Golden、ReLU `best.pt`、INT8 模型包、manifest/graph/quant 参数和 SHA-256。
- FPGA DVP/预处理/工件状态/Conv-ReLU-MaxPool/DMA/CSR RTL、仿真和 Golden 对比报告。
- HPS 图解释器、Add/Concat/Upsample、Decode/NMS、帧/tensor 管理、UART/GPIO 服务和测试。
- 空工位、正常件、四类缺陷、移动件和异常件的一条龙验收记录。
- 运行日志格式、结果协议、性能拆分、压力测试和错误统计报告。

**本次只更新本计划文件；上述所有 Create/Modify 项均是后续实施清单，不在本次创建。**
