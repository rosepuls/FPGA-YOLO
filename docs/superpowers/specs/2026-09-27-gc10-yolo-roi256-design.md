# GC10 COCO to YOLO ROI256 Dataset Design

## Purpose

Create a reproducible YOLO-format dataset for the first ZYNQ-7010 software
baseline.  The confirmed first-version input policy is **whole image as ROI**:
every source image is resized directly to 256 by 256 without letterboxing or
additional cropping.

This is a baseline dataset.  Its matching board policy is a whole OV5640 frame
as ROI (`roi_x=0`, `roi_y=0`, `roi_width=640`, `roi_height=480`).  A future
inspection-zone ROI requires a new dataset conversion and a new baseline run.

## Input and output

Input is the validated grayscale COCO dataset at
`models/data/GC10-DET-COCO`:

- `annotations/instances_{train,val,test}.json`
- `images/{train,val,test}/*.png`

Output is `models/data/GC10-DET-YOLO-ROI256`:

```text
images/train|val|test/<stem>.png
labels/train|val|test/<stem>.txt
reports/conversion_report.json
```

The output is staged in a temporary sibling directory and published only after
validation succeeds.  An existing output directory is never replaced unless
the command explicitly supplies `--overwrite`.

## Conversion rules

1. Preserve each source split exactly; no image can move between train, val,
   and test.
2. Require each source image to be single-channel grayscale and decode it with
   Pillow.
3. Resize directly from its source dimensions to `256 x 256` using the
   `RESIZE-001` bilinear rule defined in `data_contract.md`.
4. Convert COCO category ID `1..10` to YOLO class ID `0..9`.
5. Transform each COCO bbox by independent x and y scale factors, then write
   one YOLO line: `class x_center y_center width height`, normalized to
   `[0, 1]`.
6. Create an empty label file for every trusted negative image.
7. Reject malformed images, category IDs outside `1..10`, non-positive boxes,
   or transformed boxes outside the image rather than silently repairing them.

## Command-line interface

Add `models/convert_coco_to_yolo_roi256.py` with these arguments:

- `--source`, default `models/data/GC10-DET-COCO`
- `--output`, default `models/data/GC10-DET-YOLO-ROI256`
- `--overwrite`, required to replace a previous output
- `--validate <dataset-root>`, validates an existing generated dataset without
  writing

The command prints a JSON report with image/annotation counts for each split,
class counts, output location, and validation status.

## Validation and tests

Unit tests use a synthetic grayscale source dataset and prove:

- an image and its bbox are resized and normalized correctly;
- category ID 1 becomes YOLO class 0 and category ID 10 becomes class 9;
- a negative source image creates an empty label file;
- the generated dataset has paired image/label files, 256 by 256 grayscale
  images, valid normalized labels, and unchanged split membership.

An integration validation of the real output must report the expected original
split membership: 2,421 train images, 344 validation images, and 344 test
images.  The train count includes the existing augmentation and trusted
negative images.
