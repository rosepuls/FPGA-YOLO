# GC10-DET COCO Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested command-line converter under `model` that audits GC10-DET, quarantines unreliable records, creates leakage-safe splits, re-encodes accepted images, generates trustworthy negative samples, applies coordinate-preserving minority augmentation, and writes validated COCO datasets and reports.

**Architecture:** A small Python package separates source parsing, group-aware splitting, photometric augmentation, and transactional COCO writing. A thin command-line entry point exposes dry-run and conversion modes. Unit tests use synthetic temporary datasets; the final integration run audits and converts the real dataset without modifying it.

**Tech Stack:** Python 3.12, standard library, Pillow, NumPy, `unittest`.

## Global Constraints

- All executable scripts, modules, and tests live below `model`.
- The source directory `model/Data/GC10-DET` is read-only from the converter's perspective.
- Default output is `model/Data/GC10-DET-COCO`; overwrite requires `--overwrite`.
- Accepted images are single-channel 8-bit grayscale PNG files retaining original dimensions.
- Final categories are IDs 1 through 10; `10_yaozhe` and `10_yaozhed` normalize to `10_yaozhe`.
- Any unknown category quarantines the complete XML/image record.
- Physical images not consumed by exactly one valid XML record are reported but never treated as negatives.
- Split ratios default to 0.70/0.15/0.15 and grouping uses the filename material identifier.
- Additional negatives come only from trusted zero-object XML files or safe training-image background crops.
- Minority and negative augmentation is photometric only; bounding-box coordinates never change.
- Validation and test data are never augmented.
- Conversion writes to a temporary sibling directory and publishes the final directory only after validation.

---

### Task 1: Source records, label normalization, and audit

**Files:**
- Create: `model/gc10_coco/__init__.py`
- Create: `model/gc10_coco/source.py`
- Test: `model/tests/__init__.py`
- Test: `model/tests/test_source.py`

**Interfaces:**
- Produces: `Box`, `SourceRecord`, `QuarantineEntry`, `normalize_label(name)`, `material_group(stem)`, `parse_source(source_dir)`.
- `parse_source` returns `tuple[list[SourceRecord], list[QuarantineEntry], list[Path]]` containing valid records, quarantined records, and unused physical images.

- [ ] **Step 1: Write failing normalization and group tests**

```python
class SourceTests(unittest.TestCase):
    def test_label_aliases_and_unknown(self):
        self.assertEqual(normalize_label("10_yaozhed"), "10_yaozhe")
        self.assertEqual(normalize_label("10_yaozhe"), "10_yaozhe")
        with self.assertRaises(ValueError):
            normalize_label("d")

    def test_material_group_ignores_camera_and_frame(self):
        self.assertEqual(material_group("img_07_425005700_00103"), "425005700")
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python -m unittest model.tests.test_source -v`

Expected: import failure for `model.gc10_coco.source`.

- [ ] **Step 3: Implement immutable source-domain types and normalization**

```python
@dataclass(frozen=True)
class Box:
    category_id: int
    category_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

@dataclass(frozen=True)
class SourceRecord:
    xml_path: Path
    image_path: Path
    output_stem: str
    material_group: str
    width: int
    height: int
    boxes: tuple[Box, ...]
```

Implement the ten-name category map, the two class-10 aliases, and exact material-key parsing for names matching `img_<camera>_<material>_<frame>`.

- [ ] **Step 4: Write failing fixture tests for duplicate resolution and quarantine**

Create temporary numeric image folders and XML files that prove numeric `folder` wins over a duplicate filename, `all` requires a unique match, unknown labels quarantine the whole record, and unconsumed images are returned as unused.

- [ ] **Step 5: Implement XML parsing and physical-image resolution**

Validate XML size, image decode, actual size, positive in-bounds boxes, unique output stems, and known labels. Continue auditing after an individual sample fails and record a stable machine-readable reason.

- [ ] **Step 6: Run source tests**

Run: `python -m unittest model.tests.test_source -v`

Expected: all source tests pass.

- [ ] **Step 7: Commit source parsing**

```text
git add model/gc10_coco model/tests
git commit -m "feat: audit GC10 source records"
```

### Task 2: Leakage-safe grouped split

**Files:**
- Create: `model/gc10_coco/split.py`
- Test: `model/tests/test_split.py`

**Interfaces:**
- Consumes: `SourceRecord` from `model.gc10_coco.source`.
- Produces: `split_records(records, ratios, seed) -> dict[str, list[SourceRecord]]` and `summarize_splits(splits) -> dict`.

- [ ] **Step 1: Write failing split-isolation test**

```python
def test_material_group_never_crosses_splits(self):
    splits = split_records(records, (0.7, 0.15, 0.15), seed=20260925)
    owners = {}
    for split_name, members in splits.items():
        for record in members:
            previous = owners.setdefault(record.material_group, split_name)
            self.assertEqual(previous, split_name)
```

- [ ] **Step 2: Run test and verify failure**

Run: `python -m unittest model.tests.test_split -v`

Expected: import failure for `model.gc10_coco.split`.

- [ ] **Step 3: Implement deterministic greedy group allocation**

Aggregate records by material group, count images and category instances per group, sort groups by descending rarity-weighted size with seeded tie-breaking, and assign each group to the split that minimizes normalized image-count and category-count deviation from the target ratios.

- [ ] **Step 4: Add tests for determinism and bounded balance**

Run the allocator twice with the same seed and assert identical memberships. On a balanced synthetic fixture, assert each split image count differs from its target by no more than the largest group size.

- [ ] **Step 5: Run split tests**

Run: `python -m unittest model.tests.test_split -v`

Expected: all split tests pass.

- [ ] **Step 6: Commit splitting**

```text
git add model/gc10_coco/split.py model/tests/test_split.py
git commit -m "feat: add leakage-safe dataset splitting"
```

### Task 3: Coordinate-preserving augmentation and safe negatives

**Files:**
- Create: `model/gc10_coco/augment.py`
- Test: `model/tests/test_augment.py`

**Interfaces:**
- Produces: `photometric_augment(image, rng, policy) -> tuple[Image.Image, dict]`, `expanded_intersects(box, crop, margin) -> bool`, `find_safe_crop(image, boxes, rng, crop_size, margin, attempts, min_std) -> tuple[int, int, int, int] | None`.

- [ ] **Step 1: Write failing deterministic augmentation test**

```python
def test_photometric_augmentation_is_deterministic_and_same_size(self):
    first, meta1 = photometric_augment(image, random.Random(7), "minority")
    second, meta2 = photometric_augment(image, random.Random(7), "minority")
    self.assertEqual(first.size, image.size)
    self.assertEqual(first.tobytes(), second.tobytes())
    self.assertEqual(meta1, meta2)
```

- [ ] **Step 2: Run test and verify failure**

Run: `python -m unittest model.tests.test_augment -v`

Expected: import failure for `model.gc10_coco.augment`.

- [ ] **Step 3: Implement photometric policies**

Use Pillow and NumPy to implement bounded brightness/contrast, gamma LUT, Gaussian noise, weak Gaussian blur, and CLAHE-equivalent local contrast using tiled histogram equalization with clipping. The `normal` policy excludes CLAHE and uses narrower parameter ranges than `minority`.

- [ ] **Step 4: Write failing safe-crop tests**

Test that a crop intersecting an expanded box is rejected, a disjoint textured crop is accepted, a nearly uniform crop is rejected by `min_std`, and fixed seeds return identical crop coordinates.

- [ ] **Step 5: Implement safe crop selection**

Sample only fully in-bounds crops, expand every source box by the configured margin, require zero rectangle intersection, and reject crops whose grayscale standard deviation is below the configured threshold.

- [ ] **Step 6: Run augmentation tests**

Run: `python -m unittest model.tests.test_augment -v`

Expected: all augmentation tests pass.

- [ ] **Step 7: Commit augmentation**

```text
git add model/gc10_coco/augment.py model/tests/test_augment.py
git commit -m "feat: add safe photometric augmentation"
```

### Task 4: Transactional COCO writer and validation

**Files:**
- Create: `model/gc10_coco/coco.py`
- Test: `model/tests/test_coco.py`

**Interfaces:**
- Consumes: source records, split assignments, and augmentation helpers.
- Produces: `ConversionConfig`, `build_dataset(config) -> dict`, `validate_coco(dataset_root) -> dict`.

- [ ] **Step 1: Write failing Pascal VOC to COCO test**

```python
def test_box_conversion(self):
    box = Box(1, "1_chongkong", 10, 20, 31, 55)
    self.assertEqual(coco_bbox(box), [10, 20, 21, 35])
```

- [ ] **Step 2: Run test and verify failure**

Run: `python -m unittest model.tests.test_coco -v`

Expected: import failure for `model.gc10_coco.coco`.

- [ ] **Step 3: Implement original-image encoding and COCO serialization**

Re-encode each accepted source image to grayscale PNG, copy all valid annotations, generate deterministic image and annotation IDs, and write compact UTF-8 JSON with the shared ten-category list.

- [ ] **Step 4: Add minority balancing implementation and tests**

Count training instances, set the default target to the median category count, select images containing under-target categories, enforce a configurable per-source cap, call `photometric_augment(..., "minority")`, and copy every source box unchanged. Test that generated image dimensions and boxes exactly equal their sources and that no validation/test augmentation is created.

- [ ] **Step 5: Add trusted-negative generation and tests**

Retain zero-object records, generate safe 512-by-512 crops from positive training records until the requested negative ratio is reached or candidates are exhausted, optionally create mild photometric variants, and emit no annotations for them. Record failed crop attempts without treating them as conversion failures.

- [ ] **Step 6: Add transactional publishing and validation tests**

Build inside `<output>.tmp-<pid>`, validate uniqueness, category IDs, file existence, image sizes, box bounds, split group isolation, negative emptiness, and augmentation scope. Rename the temporary directory only after validation. Refuse existing output unless overwrite is explicit; with overwrite, rename the existing output to a backup, publish the new output, then remove the backup only after success.

- [ ] **Step 7: Run COCO tests**

Run: `python -m unittest model.tests.test_coco -v`

Expected: all COCO tests pass.

- [ ] **Step 8: Commit writer**

```text
git add model/gc10_coco/coco.py model/tests/test_coco.py
git commit -m "feat: write and validate COCO datasets"
```

### Task 5: Command-line entry point and end-to-end fixture

**Files:**
- Create: `model/convert_gc10_to_coco.py`
- Test: `model/tests/test_cli.py`
- Create: `model/README.md`

**Interfaces:**
- Consumes: `parse_source`, `split_records`, `build_dataset`, and `validate_coco`.
- Produces: documented `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing dry-run CLI test**

Invoke `main(["--source", fixture, "--output", output])` without `--convert`; assert return code zero, audit JSON is printed, and no output directory is created.

- [ ] **Step 2: Run CLI test and verify failure**

Run: `python -m unittest model.tests.test_cli -v`

Expected: import failure for `model.convert_gc10_to_coco`.

- [ ] **Step 3: Implement argument parsing**

Support `--source`, `--output`, `--convert`, `--overwrite`, `--seed`, `--train-ratio`, `--val-ratio`, `--test-ratio`, `--negative-ratio`, `--negative-crop-size`, `--negative-margin`, `--negative-attempts`, `--negative-min-std`, `--minority-target`, `--max-augmentations-per-image`, `--no-negative-augmentation`, and `--no-minority-augmentation`.

- [ ] **Step 4: Implement dry-run and conversion orchestration**

Dry-run parses and audits source data and prints UTF-8 JSON. Conversion builds, validates, and prints the final report. All random decisions derive from the configured seed.

- [ ] **Step 5: Write end-to-end synthetic conversion test**

Create a fixture with two production groups, valid positives, a zero-object negative, a duplicate-filename collision resolved by folder, and one unknown label. Assert the three COCO JSON files, reports, grayscale PNGs, quarantine entry, unused-image entry, and group isolation.

- [ ] **Step 6: Document exact commands**

Document dry-run, full conversion, reproducibility seed, overwrite behavior, report interpretation, and how to add genuinely verified full-frame normal images as zero-object XML records.

- [ ] **Step 7: Run the complete unit suite**

Run: `python -m unittest discover -s model/tests -v`

Expected: all tests pass.

- [ ] **Step 8: Commit CLI and documentation**

```text
git add model/convert_gc10_to_coco.py model/tests/test_cli.py model/README.md
git commit -m "feat: add GC10 COCO conversion CLI"
```

### Task 6: Real-data audit, conversion, and verification

**Files:**
- Generate: `model/Data/GC10-DET-COCO/`
- Modify only if defects are found: files created in Tasks 1 through 5 and their tests.

**Interfaces:**
- Consumes: real source dataset at `model/Data/GC10-DET`.
- Produces: validated COCO dataset and machine-readable reports.

- [ ] **Step 1: Run real-data dry-run**

Run:

```text
python model/convert_gc10_to_coco.py --source model/Data/GC10-DET --output model/Data/GC10-DET-COCO
```

Expected: 2294 XML files audited; unknown `d` record quarantined; class-10 aliases normalized; unused or ambiguous physical images reported; no output dataset created.

- [ ] **Step 2: Run full conversion**

Run:

```text
python model/convert_gc10_to_coco.py --source model/Data/GC10-DET --output model/Data/GC10-DET-COCO --convert
```

Expected: output is published only after validation and contains train, val, test images, annotations, and reports.

- [ ] **Step 3: Independently validate generated output**

Run:

```text
python model/convert_gc10_to_coco.py --validate model/Data/GC10-DET-COCO
```

Expected: exit code zero and a report with no validation errors.

- [ ] **Step 4: Run all tests again**

Run: `python -m unittest discover -s model/tests -v`

Expected: all tests pass after the real-data run.

- [ ] **Step 5: Inspect Git state and commit implementation only**

Do not add source or generated datasets. Add ignore rules for `model/Data/GC10-DET/` and `model/Data/GC10-DET-COCO/`, then commit code, tests, documentation, and ignore rules.
