from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import uuid

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLOV5_ROOT = PROJECT_ROOT / "third_party" / "yolov5"
TEST_CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("YOLOV5_CONFIG_DIR", str(TEST_CACHE_ROOT / "fpga-yolo-ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(TEST_CACHE_ROOT / "fpga-yolo-matplotlib"))
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

from utils.dataloaders import LoadImagesAndLabels, create_dataloader, read_image


HYP = {
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "flipud": 0.0,
    "fliplr": 0.0,
}


@contextmanager
def fixture_yolo_dataset():
    root = Path(__file__).parent / f"yolov5-gray-fixture-{uuid.uuid4().hex}"
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    try:
        for index, value in enumerate((32, 64, 96, 128)):
            stem = f"sample-{index}"
            Image.new("L", (256, 256), value).save(image_dir / f"{stem}.png")
            (label_dir / f"{stem}.txt").write_text(
                "0 0.500000 0.500000 0.250000 0.250000\n",
                encoding="utf-8",
            )
        yield image_dir
    finally:
        shutil.rmtree(root)


class YoloV5GrayDataloaderTests(unittest.TestCase):
    def test_read_image_preserves_singleton_channel_dimension(self) -> None:
        with fixture_yolo_dataset() as image_dir:
            image = read_image(str(next(image_dir.glob("*.png"))), channels=1)

            self.assertEqual(image.shape, (256, 256, 1))

    def test_dataset_returns_single_channel_tensor(self) -> None:
        with fixture_yolo_dataset() as image_dir:
            dataset = LoadImagesAndLabels(
                str(image_dir),
                img_size=256,
                batch_size=2,
                augment=False,
                hyp=HYP,
                channels=1,
            )

            self.assertEqual(tuple(dataset[0][0].shape), (1, 256, 256))

    def test_mosaic_and_batch_preserve_single_channel(self) -> None:
        with fixture_yolo_dataset() as image_dir:
            augmented_dataset = LoadImagesAndLabels(
                str(image_dir),
                img_size=256,
                batch_size=2,
                augment=True,
                hyp=HYP,
                channels=1,
            )
            mosaic, _ = augmented_dataset.load_mosaic(0)
            loader, _ = create_dataloader(
                str(image_dir),
                imgsz=256,
                batch_size=2,
                stride=32,
                augment=False,
                hyp=HYP,
                workers=0,
                channels=1,
            )
            batch = next(iter(loader))[0]

            self.assertEqual(mosaic.shape, (256, 256, 1))
            self.assertEqual(tuple(batch.shape), (2, 1, 256, 256))


if __name__ == "__main__":
    unittest.main()
