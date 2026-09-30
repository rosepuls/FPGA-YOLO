# LV-COCO YOLO ROI256 Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a reproducible four-class, grouped-split YOLO dataset from LV-COCO for the 256x256 grayscale FPGA path.

**Architecture:** A standalone converter reads complete records only from `train2017` and `val2017`, groups original images with filename-derived augmentations, and assigns whole groups to deterministic train/val/test splits. It writes grayscale PNGs, YOLO labels, an acceptance report, and a YOLOv5 YAML without changing LV-COCO.

**Tech Stack:** Python 3.10+, standard library, Pillow, unittest, YOLOv5 dataset YAML.

## Global Constraints

- Do not modify `data/LV-COCO` or use its incomplete `test2017` split.
- Map COCO IDs `1..4` to YOLO IDs `0..3` in ascending order.
- Accept only `RGB` and `L`; RGB conversion is `(77R + 150G + 29B + 128) >> 8`.
- Emit `256 x 256` `L`-mode PNGs and same-stem YOLO `.txt` files.
- Use a fixed seed and keep every `_aug_<integer>` sibling in one split.
- Reject missing files, duplicate IDs, invalid boxes, unknown modes, and missing per-class split coverage.
- Publish via an atomic same-parent temporary directory; replacement requires `--overwrite`.

---

## File Structure

- Create `models/convert_lv_coco_to_yolo_roi256.py`: validation, grouping, allocation, conversion, output audit, CLI.
- Create `models/tests/test_lv_yolo_roi256.py`: fixture-driven unit and conversion tests.
- Create `lv_yolo.yaml`: YOLOv5 dataset entry point.
- Generate `data/LV-YOLO-ROI256/`: images, labels, and `reports/conversion_report.json`.

### Task 1: Read and validate usable LV source records

**Files:**
- Create: `models/convert_lv_coco_to_yolo_roi256.py`
- Create: `models/tests/test_lv_yolo_roi256.py`

**Interfaces:**
- Produces `SourceRecord(source_path, file_name, width, height, annotations, group_key)`.
- Produces `load_source_records(source_dir: Path) -> tuple[list[SourceRecord], dict[int, str]]`.

- [ ] **Step 1: Write failing tests**

```python
def test_load_source_records_merges_train_val_and_groups_augmented_files(tmp_path):
    source = make_lv_source(tmp_path, {"train": ["part.jpg", "part_aug_7.jpg"], "val": ["other.jpg"]})
    records, categories = load_source_records(source)
    assert len(records) == 3
    assert {record.group_key for record in records} == {"part", "other"}
    assert tuple(categories) == (1, 2, 3, 4)

def test_load_source_records_rejects_missing_referenced_image(tmp_path):
    source = make_lv_source(tmp_path, {"train": ["missing.jpg"], "val": []}, create_images=False)
    with self.assertRaisesRegex(ValueError, "missing source image"):
        load_source_records(source)
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest models.tests.test_lv_yolo_roi256`

Expected: FAIL because `convert_lv_coco_to_yolo_roi256` is absent.

- [ ] **Step 3: Implement the smallest loader**

```python
def group_key(file_name: str) -> str:
    return re.sub(r"_aug_\d+$", "", Path(file_name).stem)

def load_source_records(source_dir: Path) -> tuple[list[SourceRecord], dict[int, str]]:
    records, categories = [], None
    for split_name in ("train", "val"):
        split_records, split_categories = _read_valid_split(source_dir, split_name)
        records.extend(split_records)
        categories = split_categories if categories is None else _require_same_categories(categories, split_categories)
    return records, categories
```

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest models.tests.test_lv_yolo_roi256`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/convert_lv_coco_to_yolo_roi256.py models/tests/test_lv_yolo_roi256.py
git commit -m "feat: validate LV COCO source records"
```

### Task 2: Preserve augmentation groups and convert primitives

**Files:**
- Modify: `models/convert_lv_coco_to_yolo_roi256.py`
- Modify: `models/tests/test_lv_yolo_roi256.py`

**Interfaces:**
- Consumes `list[SourceRecord]`.
- Produces `assign_splits(records: list[SourceRecord], seed: int) -> dict[str, list[SourceRecord]]`.
- Produces `gray_u8(image: Image.Image) -> Image.Image` and `format_yolo_label(category_id, bbox, width, height) -> str`.

- [ ] **Step 1: Write failing tests**

```python
def test_assign_splits_is_repeatable_and_never_splits_augmented_group(self):
    records = fixture_records_with_all_categories_and_augmented_group()
    first = assign_splits(records, seed=20260928)
    second = assign_splits(records, seed=20260928)
    self.assertEqual(split_group_map(first), split_group_map(second))
    assert_group_keys_are_disjoint(first)
    assert_all_categories_present(first)

def test_gray_u8_uses_fixed_point_rgb_formula(self):
    image = Image.new("RGB", (1, 1), (1, 2, 3))
    self.assertEqual(list(gray_u8(image).getdata()), [(77 + 300 + 87 + 128) >> 8])
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest models.tests.test_lv_yolo_roi256`

Expected: FAIL because allocator and primitives are absent.

- [ ] **Step 3: Implement deterministic allocation and fixed-point grayscale**

```python
def assign_splits(records: list[SourceRecord], seed: int) -> dict[str, list[SourceRecord]]:
    grouped_records = _group_records(records)
    assigned = _allocate_groups_to_class_targets(grouped_records, seed, ratios=(0.8, 0.1, 0.1))
    _require_group_disjoint_class_coverage(assigned)
    return assigned

def gray_u8(image: Image.Image) -> Image.Image:
    if image.mode == "L":
        return image.copy()
    if image.mode != "RGB":
        raise ValueError(f"unsupported source image mode: {image.mode}")
    return Image.frombytes("L", image.size, bytes(
        (77 * r + 150 * g + 29 * b + 128) >> 8 for r, g, b in image.getdata()
    ))
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m unittest models.tests.test_lv_yolo_roi256`

Expected: PASS.

```bash
git add models/convert_lv_coco_to_yolo_roi256.py models/tests/test_lv_yolo_roi256.py
git commit -m "feat: split LV records by augmentation group"
```

### Task 3: Write, validate, and configure the data artifact

**Files:**
- Modify: `models/convert_lv_coco_to_yolo_roi256.py`
- Modify: `models/tests/test_lv_yolo_roi256.py`
- Create: `lv_yolo.yaml`

**Interfaces:**
- Produces `convert_dataset(config: ConversionConfig) -> dict[str, object]`.
- Produces `validate_dataset(root: Path) -> dict[str, object]`.

- [ ] **Step 1: Write failing tests**

```python
def test_convert_dataset_writes_valid_l_pngs_paired_labels_and_report(self, tmp_path):
    output = tmp_path / "LV-YOLO-ROI256"
    report = convert_dataset(ConversionConfig(make_complete_four_class_lv_source(tmp_path), output, seed=20260928))
    self.assertTrue(report["validation"]["valid"])
    assert_output_images_are_l_mode_256_pngs(output)
    self.assertTrue((output / "reports" / "conversion_report.json").is_file())

def test_convert_dataset_refuses_existing_output_without_overwrite(self, tmp_path):
    output = tmp_path / "LV-YOLO-ROI256"
    config = ConversionConfig(make_complete_four_class_lv_source(tmp_path), output, seed=20260928)
    convert_dataset(config)
    with self.assertRaises(FileExistsError):
        convert_dataset(config)
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest models.tests.test_lv_yolo_roi256`

Expected: FAIL because dataset writer is absent.

- [ ] **Step 3: Implement atomic writer and validator**

```python
def convert_dataset(config: ConversionConfig) -> dict[str, object]:
    records, categories = load_source_records(config.source_dir)
    assignments = assign_splits(records, config.seed)
    return _write_validate_and_publish(config, assignments, categories)

def validate_dataset(root: Path) -> dict[str, object]:
    result = _validate_output_files(root)
    return _validate_reported_group_disjointness(root, result)
```

Create `lv_yolo.yaml` with `path: ../../data/LV-YOLO-ROI256` (relative to `third_party/yolov5`), three `images/*` splits, `nc: 4`, and names `[ca_shang, zang_wu, zhe_zhou, zhen_kong]`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m unittest models.tests.test_lv_yolo_roi256 -v`

Expected: PASS with no errors or warnings.

```bash
git add models/convert_lv_coco_to_yolo_roi256.py models/tests/test_lv_yolo_roi256.py lv_yolo.yaml
git commit -m "feat: convert LV COCO data to YOLO ROI256"
```

### Task 4: Generate and independently accept the LV dataset

**Files:**
- Create: `data/LV-YOLO-ROI256/images/{train,val,test}/`
- Create: `data/LV-YOLO-ROI256/labels/{train,val,test}/`
- Create: `data/LV-YOLO-ROI256/reports/conversion_report.json`

**Interfaces:**
- Consumes Task 3 CLI and `data/LV-COCO`.
- Produces an audited YOLOv5-ready data artifact.

- [ ] **Step 1: Generate**

```bash
python models/convert_lv_coco_to_yolo_roi256.py --source data/LV-COCO --output data/LV-YOLO-ROI256 --seed 20260928
```

Expected: a report with `validation.valid: true`.

- [ ] **Step 2: Independently validate**

```bash
python models/convert_lv_coco_to_yolo_roi256.py --validate data/LV-YOLO-ROI256
```

Expected: valid image/label pairs, valid geometry, classes `0..3`, and group-disjoint splits.

- [ ] **Step 3: Commit the configuration and report**

```bash
git add lv_yolo.yaml data/LV-YOLO-ROI256/reports/conversion_report.json
git commit -m "data: add LV YOLO ROI256 conversion report"
```

## Plan Self-Review

- Tasks 1–2 cover source integrity, class mapping, grouping, grayscale conversion, and labels.
- Task 3 covers atomic output, output validation, and YAML configuration.
- Task 4 covers generation and a CLI acceptance gate.
- Interfaces flow from `SourceRecord` to split allocation to writer/validator; no deferred or ambiguous behavior remains.
