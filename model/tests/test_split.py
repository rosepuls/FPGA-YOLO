import unittest
from pathlib import Path

from model.gc10_coco.source import Box, SourceRecord
from model.gc10_coco.split import split_records, summarize_splits


def record(index: int, group: str, categories: tuple[int, ...]) -> SourceRecord:
    boxes = tuple(Box(category, f"class-{category}", 1, 1, 5, 5) for category in categories)
    return SourceRecord(
        xml_path=Path(f"{index}.xml"),
        image_path=Path(f"{index}.jpg"),
        output_stem=f"record-{index}",
        material_group=group,
        width=32,
        height=24,
        boxes=boxes,
    )


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            record(1, "A", (1, 1)),
            record(2, "A", (2,)),
            record(3, "B", (1,)),
            record(4, "C", (2, 2)),
            record(5, "D", (3,)),
            record(6, "E", (3, 3)),
            record(7, "F", (1, 3)),
            record(8, "G", (2,)),
        ]

    def test_material_group_never_crosses_splits(self) -> None:
        splits = split_records(self.records, (0.5, 0.25, 0.25), seed=7)
        owners: dict[str, str] = {}
        for split_name, members in splits.items():
            for item in members:
                previous = owners.setdefault(item.material_group, split_name)
                self.assertEqual(previous, split_name)
        self.assertEqual(sum(len(items) for items in splits.values()), len(self.records))

    def test_same_seed_produces_identical_memberships(self) -> None:
        first = split_records(self.records, (0.5, 0.25, 0.25), seed=19)
        second = split_records(list(reversed(self.records)), (0.5, 0.25, 0.25), seed=19)
        first_names = {name: [item.output_stem for item in values] for name, values in first.items()}
        second_names = {name: [item.output_stem for item in values] for name, values in second.items()}
        self.assertEqual(first_names, second_names)

    def test_summary_counts_images_groups_and_instances(self) -> None:
        splits = split_records(self.records, (0.5, 0.25, 0.25), seed=3)
        summary = summarize_splits(splits)
        self.assertEqual(sum(item["images"] for item in summary.values()), 8)
        self.assertEqual(sum(item["groups"] for item in summary.values()), 7)
        self.assertEqual(sum(sum(item["instances"].values()) for item in summary.values()), 12)

    def test_invalid_ratios_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_records(self.records, (0.8, 0.3, -0.1), seed=1)

    def test_image_ratio_is_primary_when_large_groups_are_class_imbalanced(self) -> None:
        records: list[SourceRecord] = []
        index = 0
        for group, size, category in (("A", 20, 5), ("B", 15, 6), ("C", 15, 6)):
            for _ in range(size):
                index += 1
                records.append(record(index, group, (category,)))
        for singleton in range(50):
            index += 1
            records.append(record(index, f"S{singleton}", ((singleton % 10) + 1,)))

        splits = split_records(records, (0.7, 0.15, 0.15), seed=1)

        self.assertEqual({name: len(items) for name, items in splits.items()}, {"train": 70, "val": 15, "test": 15})


if __name__ == "__main__":
    unittest.main()
