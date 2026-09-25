from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


CATEGORY_NAMES = (
    "1_chongkong",
    "2_hanfeng",
    "3_yueyawan",
    "4_shuiban",
    "5_youban",
    "6_siban",
    "7_yiwu",
    "8_yahen",
    "9_zhehen",
    "10_yaozhe",
)

_CATEGORY_IDS = {name: index for index, name in enumerate(CATEGORY_NAMES, start=1)}
_LABEL_ALIASES = {"10_yaozhed": "10_yaozhe"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


@dataclass(frozen=True)
class QuarantineEntry:
    xml_path: Path
    filename: str
    reason: str
    detail: str


def normalize_label(name: str) -> tuple[int, str]:
    normalized = _LABEL_ALIASES.get(name.strip(), name.strip())
    if normalized not in _CATEGORY_IDS:
        raise ValueError(f"unknown category: {name!r}")
    return _CATEGORY_IDS[normalized], normalized


def material_group(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] != "img" or not parts[1] or not parts[2] or not parts[3]:
        raise ValueError(f"unexpected image name: {stem!r}")
    return parts[2]


def _image_files(source_dir: Path) -> list[Path]:
    result: list[Path] = []
    for directory in sorted(source_dir.iterdir()):
        if not directory.is_dir() or not directory.name.isdigit():
            continue
        result.extend(
            path
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
    return result


def _required_text(root: ET.Element, path: str) -> str:
    value = root.findtext(path)
    if value is None or not value.strip():
        raise ValueError(f"missing XML field: {path}")
    return value.strip()


def _quarantine(xml_path: Path, filename: str, reason: str, detail: str) -> QuarantineEntry:
    return QuarantineEntry(xml_path=xml_path, filename=filename, reason=reason, detail=detail)


def parse_source(source_dir: str | Path) -> tuple[list[SourceRecord], list[QuarantineEntry], list[Path]]:
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset directory does not exist: {source}")
    label_dir = source / "lable"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"XML directory does not exist: {label_dir}")

    all_images = _image_files(source)
    images_by_name: dict[str, list[Path]] = {}
    for image_path in all_images:
        images_by_name.setdefault(image_path.name, []).append(image_path)

    records: list[SourceRecord] = []
    quarantined: list[QuarantineEntry] = []
    consumed: set[Path] = set()
    output_stems: set[str] = set()

    for xml_path in sorted(label_dir.glob("*.xml")):
        filename = ""
        image_path: Path | None = None
        try:
            root = ET.parse(xml_path).getroot()
            folder = _required_text(root, "folder")
            filename = _required_text(root, "filename")
            candidates = images_by_name.get(filename, [])
            if folder.isdigit():
                requested = source / folder / filename
                if requested not in candidates or not requested.is_file():
                    raise LookupError("missing_image")
                image_path = requested
            elif folder == "all":
                if len(candidates) != 1:
                    raise LookupError("ambiguous_image" if candidates else "missing_image")
                image_path = candidates[0]
            else:
                raise LookupError("unsupported_folder")

            if image_path in consumed:
                raise LookupError("duplicate_source_image")

            declared_width = int(_required_text(root, "size/width"))
            declared_height = int(_required_text(root, "size/height"))
            with Image.open(image_path) as image:
                image.load()
                actual_width, actual_height = image.size
            if (actual_width, actual_height) != (declared_width, declared_height):
                raise ArithmeticError(
                    f"declared {declared_width}x{declared_height}, actual {actual_width}x{actual_height}"
                )

            boxes: list[Box] = []
            for object_node in root.findall("object"):
                category_id, category_name = normalize_label(_required_text(object_node, "name"))
                box_node = object_node.find("bndbox")
                if box_node is None:
                    raise ValueError("missing bndbox")
                xmin = int(float(_required_text(box_node, "xmin")))
                ymin = int(float(_required_text(box_node, "ymin")))
                xmax = int(float(_required_text(box_node, "xmax")))
                ymax = int(float(_required_text(box_node, "ymax")))
                if not (0 <= xmin < xmax <= actual_width and 0 <= ymin < ymax <= actual_height):
                    raise ArithmeticError(
                        f"invalid bbox {(xmin, ymin, xmax, ymax)} for {actual_width}x{actual_height}"
                    )
                boxes.append(Box(category_id, category_name, xmin, ymin, xmax, ymax))

            output_stem = f"{image_path.parent.name}__{image_path.stem}"
            if output_stem in output_stems:
                raise LookupError("duplicate_output_name")
            group = material_group(image_path.stem)
            records.append(
                SourceRecord(
                    xml_path=xml_path,
                    image_path=image_path,
                    output_stem=output_stem,
                    material_group=group,
                    width=actual_width,
                    height=actual_height,
                    boxes=tuple(boxes),
                )
            )
            consumed.add(image_path)
            output_stems.add(output_stem)
        except ET.ParseError as error:
            quarantined.append(_quarantine(xml_path, filename, "invalid_xml", str(error)))
        except LookupError as error:
            quarantined.append(_quarantine(xml_path, filename, str(error), str(error)))
        except (OSError, ValueError, ArithmeticError) as error:
            message = str(error)
            if message.startswith("unknown category"):
                reason = "unknown_label"
            elif message.startswith("invalid bbox"):
                reason = "invalid_bbox"
            elif message.startswith("declared"):
                reason = "size_mismatch"
            elif isinstance(error, OSError):
                reason = "unreadable_image"
            else:
                reason = "invalid_record"
            quarantined.append(_quarantine(xml_path, filename, reason, message))

    unused = sorted((path for path in all_images if path not in consumed), key=lambda path: str(path))
    return records, quarantined, unused
