import random
import unittest

import numpy as np
from PIL import Image

from model.gc10_coco.augment import expanded_intersects, find_safe_crop, photometric_augment
from model.gc10_coco.source import Box


class AugmentTests(unittest.TestCase):
    def setUp(self) -> None:
        values = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) % 256
        self.image = Image.fromarray(values.astype(np.uint8), mode="L")

    def test_photometric_augmentation_is_deterministic_and_same_size(self) -> None:
        first, metadata1 = photometric_augment(self.image, random.Random(7), "minority")
        second, metadata2 = photometric_augment(self.image, random.Random(7), "minority")
        self.assertEqual(first.mode, "L")
        self.assertEqual(first.size, self.image.size)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(metadata1, metadata2)

    def test_normal_policy_uses_only_declared_mild_operations(self) -> None:
        _, metadata = photometric_augment(self.image, random.Random(3), "normal")
        self.assertIn(metadata["operation"], {"brightness_contrast", "gamma", "noise", "blur"})

    def test_expanded_box_rejects_touching_crop(self) -> None:
        box = Box(1, "1_chongkong", 10, 10, 20, 20)
        self.assertTrue(expanded_intersects(box, (4, 4, 9, 9), margin=2))
        self.assertFalse(expanded_intersects(box, (0, 0, 7, 7), margin=2))

    def test_safe_crop_is_textured_and_disjoint(self) -> None:
        box = Box(1, "1_chongkong", 10, 10, 22, 22)
        crop = find_safe_crop(
            self.image,
            (box,),
            random.Random(11),
            crop_size=8,
            margin=2,
            attempts=200,
            min_std=5.0,
        )
        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertFalse(expanded_intersects(box, crop, margin=2))
        self.assertGreaterEqual(np.asarray(self.image.crop(crop)).std(), 5.0)

    def test_uniform_image_has_no_safe_textured_crop(self) -> None:
        uniform = Image.new("L", (32, 32), 128)
        crop = find_safe_crop(uniform, (), random.Random(1), 8, 0, 20, min_std=1.0)
        self.assertIsNone(crop)


if __name__ == "__main__":
    unittest.main()
