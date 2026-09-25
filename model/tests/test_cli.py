from contextlib import contextmanager, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import unittest
import uuid

from PIL import Image

from model.convert_gc10_to_coco import main
from model.tests.test_source import write_xml


@contextmanager
def test_directory():
    path = Path(__file__).parent / f"cli-fixture-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


class CliTests(unittest.TestCase):
    def test_dry_run_prints_audit_without_creating_output(self) -> None:
        with test_directory() as root:
            source = root / "GC10-DET"
            (source / "1").mkdir(parents=True)
            (source / "lable").mkdir()
            filename = "img_01_BATCH1_00001.jpg"
            Image.new("L", (32, 24), 100).save(source / "1" / filename)
            write_xml(source / "lable" / "sample.xml", "1", filename, [("1_chongkong", (2, 3, 20, 18))])
            output = root / "out"
            stdout = StringIO()

            with redirect_stdout(stdout):
                result = main(["--source", str(source), "--output", str(output)])

            report = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(report["accepted_source_images"], 1)
            self.assertFalse(output.exists())

    def test_convert_then_validate_modes_succeed(self) -> None:
        with test_directory() as root:
            source = root / "GC10-DET"
            (source / "1").mkdir(parents=True)
            (source / "lable").mkdir()
            filename = "img_01_BATCH2_00001.jpg"
            Image.new("L", (32, 24), 100).save(source / "1" / filename)
            write_xml(source / "lable" / "sample.xml", "1", filename, [("1_chongkong", (2, 3, 20, 18))])
            output = root / "out"

            with redirect_stdout(StringIO()):
                converted = main(
                    [
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                        "--convert",
                        "--no-negative-augmentation",
                        "--no-minority-augmentation",
                    ]
                )
            validation_stdout = StringIO()
            with redirect_stdout(validation_stdout):
                validated = main(["--validate", str(output)])

            self.assertEqual(converted, 0)
            self.assertEqual(validated, 0)
            self.assertTrue(json.loads(validation_stdout.getvalue())["valid"])


if __name__ == "__main__":
    unittest.main()
