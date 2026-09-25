from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Iterable

from .source import SourceRecord


SPLIT_NAMES = ("train", "val", "test")


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("split ratios must contain three positive values")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")


def _instance_counts(records: Iterable[SourceRecord]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for item in records:
        counts.update(box.category_id for box in item.boxes)
    return counts


def split_records(
    records: list[SourceRecord],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 20260925,
) -> dict[str, list[SourceRecord]]:
    _validate_ratios(ratios)
    if not records:
        return {name: [] for name in SPLIT_NAMES}

    grouped: dict[str, list[SourceRecord]] = defaultdict(list)
    for item in sorted(records, key=lambda value: value.output_stem):
        grouped[item.material_group].append(item)

    global_instances = _instance_counts(records)
    categories = sorted(global_instances)
    randomizer = random.Random(seed)
    tie_break = {group: randomizer.random() for group in sorted(grouped)}

    def priority(entry: tuple[str, list[SourceRecord]]) -> tuple[float, int, float, str]:
        group, members = entry
        counts = _instance_counts(members)
        rarity = sum(count / max(global_instances[category], 1) for category, count in counts.items())
        return (-rarity, -len(members), tie_break[group], group)

    ordered_groups = sorted(grouped.items(), key=priority)
    target_images = {name: len(records) * ratio for name, ratio in zip(SPLIT_NAMES, ratios)}
    target_instances = {
        name: {category: global_instances[category] * ratio for category in categories}
        for name, ratio in zip(SPLIT_NAMES, ratios)
    }
    assignments: dict[str, list[SourceRecord]] = {name: [] for name in SPLIT_NAMES}
    current_images = Counter({name: 0 for name in SPLIT_NAMES})
    current_instances: dict[str, Counter[int]] = {name: Counter() for name in SPLIT_NAMES}

    for group, members in ordered_groups:
        group_counts = _instance_counts(members)
        candidate_scores: list[tuple[float, float, int, str]] = []
        for split_index, split_name in enumerate(SPLIT_NAMES):
            projected_images = current_images[split_name] + len(members)
            image_error = ((projected_images - target_images[split_name]) / max(target_images[split_name], 1.0)) ** 2
            class_error = 0.0
            for category in categories:
                projected = current_instances[split_name][category] + group_counts[category]
                target = target_instances[split_name][category]
                class_error += ((projected - target) / max(target, 1.0)) ** 2
            fill = current_images[split_name] / max(target_images[split_name], 1.0)
            candidate_scores.append((image_error + class_error / max(len(categories), 1), fill, split_index, split_name))
        chosen = min(candidate_scores)[3]
        assignments[chosen].extend(members)
        current_images[chosen] += len(members)
        current_instances[chosen].update(group_counts)

    for members in assignments.values():
        members.sort(key=lambda item: item.output_stem)
    return assignments


def summarize_splits(splits: dict[str, list[SourceRecord]]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for split_name in SPLIT_NAMES:
        records = splits.get(split_name, [])
        counts = _instance_counts(records)
        summary[split_name] = {
            "images": len(records),
            "groups": len({item.material_group for item in records}),
            "instances": {str(category): counts[category] for category in sorted(counts)},
        }
    return summary
