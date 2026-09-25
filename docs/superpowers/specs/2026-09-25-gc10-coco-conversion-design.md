# GC10-DET to COCO conversion and augmentation design

## Goal

Convert the source GC10-DET dataset into a reproducible COCO detection dataset without modifying the source files. The conversion must quarantine unreliable samples, normalize labels and image encoding, prevent sequence leakage across splits, create trustworthy negative samples, and improve minority-class representation using coordinate-preserving photometric augmentation.

## Source and output

Source dataset:

```text
model/Data/GC10-DET
```

Default output dataset:

```text
model/Data/GC10-DET-COCO
```

The converter never deletes or edits source images or XML files. The output directory is created only when it does not already exist. Replacing an existing output requires an explicit command-line option.

## Category mapping

The final dataset contains ten categories with stable COCO category IDs from 1 through 10:

```text
1_chongkong
2_hanfeng
3_yueyawan
4_shuiban
5_youban
6_siban
7_yiwu
8_yahen
9_zhehen
10_yaozhe
```

Both `10_yaozhe` and `10_yaozhed` map to `10_yaozhe`. Any other unknown category is an error. The sample containing category `d` is quarantined as a whole because removing only that annotation could leave a real but unlabeled defect in a training image.

## Source image resolution

For XML records whose `folder` is a numeric class directory, the converter resolves the image using `folder/filename`. For XML records whose folder is `all`, it resolves the filename only when exactly one physical image matches. Ambiguous, missing, unreadable, size-mismatched, or invalidly annotated records are quarantined.

Physical images not consumed by one valid XML record are reported as unused and are not treated as normal samples. This prevents the known same-filename/different-content collisions from silently overwriting one another.

## Image normalization and naming

Every accepted image is decoded, converted to single-channel 8-bit grayscale, and saved losslessly as PNG. The output filename contains the resolved source directory and original stem, for example:

```text
2__img_03_425506300_00018.png
```

This provides stable unique names and removes dependency on original JPEG decoder metadata. Original width and height are retained. Bounding boxes remain in original image coordinates.

## Dataset split

The split is deterministic with a configurable random seed and default ratios of 70% train, 15% validation, and 15% test.

The grouping key is the production/material identifier in the filename, such as `425005700` in `img_01_425005700_00103`. All cameras and frames sharing that identifier stay in the same split. This is deliberately stricter than grouping by the full camera prefix and prevents closely related production sequences from leaking into validation or test data.

A deterministic greedy group allocator balances both image counts and category-instance counts against the requested split ratios. The audit report records the achieved ratios and category distributions.

## Normal samples

Trusted zero-object XML images are retained as negative images with no COCO annotations.

Additional negative images are generated only from the training split. The converter samples fixed-size square crops from annotated source images and accepts a crop only when:

- it has zero intersection with every ground-truth box expanded by a configurable safety margin;
- it lies fully inside the source image;
- its intensity standard deviation exceeds a small threshold, rejecting nearly blank crops;
- its source group is already assigned to the training split.

Default negative crops are 512 by 512 pixels and are re-encoded as grayscale PNG. They have empty annotation lists. The default target is one negative image for every four positive training images, including existing trusted negatives. The target and crop size are configurable.

Negative samples receive a dedicated mild photometric augmentation policy: small brightness and contrast changes, gamma adjustment, low-amplitude Gaussian noise, and optional weak blur. No geometric transformation is applied.

## Minority-class augmentation

Minority augmentation is applied only after splitting and only to training images. Validation and test images are never augmented.

By default, a category is considered a minority category when its training instance count is below the median training category count. The target is the median count, configurable from the command line.

The converter repeatedly selects training images containing under-target categories. Each selected image is transformed using one deterministic, randomly parameterized photometric pipeline chosen from:

- brightness and contrast adjustment;
- gamma adjustment;
- low-amplitude Gaussian noise;
- weak Gaussian blur;
- CLAHE.

The image dimensions, pixel positions, and all bounding boxes remain unchanged. If a selected image contains several classes, all annotations are copied. Generation stops when all minority classes reach the target or a bounded augmentation limit is reached. The manifest records each augmented image, its source, seed, transformation parameters, and copied annotations.

## COCO output

The generated layout is:

```text
GC10-DET-COCO/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
└── reports/
    ├── audit_report.json
    ├── quarantine.json
    ├── unused_images.json
    ├── split_manifest.json
    └── augmentation_manifest.json
```

COCO bounding boxes use `[x, y, width, height]`. Annotation area is `width * height`, `iscrowd` is zero, and IDs are deterministic positive integers.

## Command-line interface

The converter provides a dry-run audit mode and a conversion mode. Important configurable values include:

- source and output directories;
- random seed and split ratios;
- overwrite permission;
- negative crop size, safety margin, and target ratio;
- minority threshold and target count;
- maximum augmentations per source image;
- optional augmentation disable switches.

Dry-run is the default safe behavior unless an explicit conversion flag is provided.

## Failure handling

The converter fails before writing output when category mappings, split ratios, or required source directories are invalid. Individual bad samples are quarantined with a reason rather than terminating the complete conversion. Any partially generated output is written to a temporary sibling directory and renamed to the final destination only after all validation passes.

The final validation checks:

- every COCO image exists and is readable;
- image IDs, annotation IDs, and filenames are unique;
- all category IDs are valid;
- all bounding boxes are positive and inside image bounds;
- all source production groups occur in one split only;
- validation and test contain no augmented samples;
- negative samples contain no annotations;
- augmentation records preserve source image dimensions and box coordinates.

## Tests

Automated tests cover:

- label normalization and rejection of unknown labels;
- duplicate-filename resolution using the XML folder;
- quarantine behavior for ambiguous and missing images;
- Pascal VOC to COCO bounding-box conversion;
- production-group extraction and split isolation;
- safe negative-crop intersection checks;
- deterministic photometric augmentation;
- coordinate preservation for minority augmentation;
- COCO ID uniqueness and bounds validation;
- end-to-end conversion of a small synthetic fixture.

## Non-goals

This converter does not fabricate full-frame normal images by inpainting defects, does not treat unlabeled images as negatives, does not apply geometric augmentation, and does not resize all source images to 256 by 256. Model-specific tiling, letterboxing, and tensor normalization remain part of the later training input pipeline so they can be matched exactly to the FPGA ROI and resize implementation.
