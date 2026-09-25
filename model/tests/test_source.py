from contextlib import contextmanager
import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image

from model.gc10_coco.source import material_group, normalize_label, parse_source


TEST_TEMP_ROOT = Path(__file__).parent


@contextmanager
def test_directory():
    path = TEST_TEMP_ROOT / f"fixture-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_xml(path: Path, folder: str, filename: str, objects: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    object_xml = "".join(
        f"""
        <object><name>{name}</name><bndbox>
        <xmin>{box[0]}</xmin><ymin>{box[1]}</ymin>
        <xmax>{box[2]}</xmax><ymax>{box[3]}</ymax>
        </bndbox></object>
        """
        for name, box in objects
    )
    path.write_text(
        f"""<annotation><folder>{folder}</folder><filename>{filename}</filename>
        <size><width>32</width><height>24</height><depth>1</depth></size>
        {object_xml}</annotation>""",
        encoding="utf-8",
    )


class SourceTests(unittest.TestCase):
    def test_label_aliases_and_unknown(self) -> None:
        self.assertEqual(normalize_label("10_yaozhed"), (10, "10_yaozhe"))
        self.assertEqual(normalize_label("10_yaozhe"), (10, "10_yaozhe"))
        with self.assertRaises(ValueError):
            normalize_label("d")

    def test_material_group_ignores_camera_and_frame(self) -> None:
        self.assertEqual(material_group("img_07_425005700_00103"), "425005700")
        with self.assertRaises(ValueError):
            material_group("bad-name")

    def test_numeric_folder_resolves_duplicate_filename_and_reports_unused(self) -> None:
        with test_directory() as root:
            (root / "1").mkdir()
            (root / "2").mkdir()
            (root / "lable").mkdir()
            name = "img_01_BATCH1_00001.jpg"
            Image.new("L", (32, 24), 20).save(root / "1" / name)
            Image.new("L", (32, 24), 220).save(root / "2" / name)
            write_xml(root / "lable" / "sample.xml", "2", name, [("2_hanfeng", (2, 3, 20, 18))])

            records, quarantine, unused = parse_source(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].image_path, root / "2" / name)
            self.assertEqual(records[0].output_stem, "2__img_01_BATCH1_00001")
            self.assertEqual(quarantine, [])
            self.assertEqual(unused, [root / "1" / name])

    def test_all_folder_with_duplicate_filename_is_quarantined(self) -> None:
        with test_directory() as root:
            (root / "1").mkdir()
            (root / "2").mkdir()
            (root / "lable").mkdir()
            name = "img_01_BATCH2_00001.jpg"
            Image.new("L", (32, 24), 20).save(root / "1" / name)
            Image.new("L", (32, 24), 220).save(root / "2" / name)
            write_xml(root / "lable" / "sample.xml", "all", name, [("1_chongkong", (2, 3, 20, 18))])

            records, quarantine, unused = parse_source(root)

            self.assertEqual(records, [])
            self.assertEqual(quarantine[0].reason, "ambiguous_image")
            self.assertEqual(len(unused), 2)

    def test_unknown_label_quarantines_complete_record(self) -> None:
        with test_directory() as root:
            (root / "2").mkdir()
            (root / "lable").mkdir()
            name = "img_01_BATCH3_00001.jpg"
            Image.new("L", (32, 24), 100).save(root / "2" / name)
            write_xml(
                root / "lable" / "sample.xml",
                "2",
                name,
                [("2_hanfeng", (2, 3, 20, 18)), ("d", (21, 4, 30, 12))],
            )

            records, quarantine, unused = parse_source(root)

            self.assertEqual(records, [])
            self.assertEqual(quarantine[0].reason, "unknown_label")
            self.assertEqual(unused, [root / "2" / name])


if __name__ == "__main__":
    unittest.main()
