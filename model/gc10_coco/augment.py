from __future__ import annotations

import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .source import Box


def _gamma(image: Image.Image, value: float) -> Image.Image:
    inverse = 1.0 / value
    table = [round(((level / 255.0) ** inverse) * 255.0) for level in range(256)]
    return image.point(table)


def _noise(image: Image.Image, rng: random.Random, sigma: float) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    generator = np.random.default_rng(rng.getrandbits(64))
    noisy = np.clip(array + generator.normal(0.0, sigma, array.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def _clahe_like(image: Image.Image, tiles: int = 8, clip_factor: float = 2.0) -> Image.Image:
    source = np.asarray(image, dtype=np.uint8)
    height, width = source.shape
    result = np.empty_like(source)
    y_edges = np.linspace(0, height, min(tiles, height) + 1, dtype=int)
    x_edges = np.linspace(0, width, min(tiles, width) + 1, dtype=int)
    for y_index in range(len(y_edges) - 1):
        for x_index in range(len(x_edges) - 1):
            y0, y1 = y_edges[y_index], y_edges[y_index + 1]
            x0, x1 = x_edges[x_index], x_edges[x_index + 1]
            tile = source[y0:y1, x0:x1]
            histogram = np.bincount(tile.ravel(), minlength=256).astype(np.int64)
            limit = max(1, int(clip_factor * tile.size / 256))
            excess = int(np.maximum(histogram - limit, 0).sum())
            histogram = np.minimum(histogram, limit)
            histogram += excess // 256
            histogram[: excess % 256] += 1
            cumulative = histogram.cumsum()
            nonzero = cumulative[cumulative > 0]
            if nonzero.size == 0 or cumulative[-1] == nonzero[0]:
                result[y0:y1, x0:x1] = tile
                continue
            lookup = np.clip(
                (cumulative - nonzero[0]) * 255.0 / (cumulative[-1] - nonzero[0]),
                0,
                255,
            ).astype(np.uint8)
            result[y0:y1, x0:x1] = lookup[tile]
    return Image.fromarray(result)


def photometric_augment(
    image: Image.Image,
    rng: random.Random,
    policy: str,
) -> tuple[Image.Image, dict]:
    grayscale = image.convert("L")
    if policy == "normal":
        operations = ("brightness_contrast", "gamma", "noise", "blur")
    elif policy == "minority":
        operations = ("brightness_contrast", "gamma", "noise", "blur", "clahe")
    else:
        raise ValueError(f"unknown augmentation policy: {policy}")

    operation = rng.choice(operations)
    if operation == "brightness_contrast":
        spread = 0.10 if policy == "normal" else 0.20
        brightness = rng.uniform(1.0 - spread, 1.0 + spread)
        contrast = rng.uniform(1.0 - spread, 1.0 + spread)
        transformed = ImageEnhance.Brightness(grayscale).enhance(brightness)
        transformed = ImageEnhance.Contrast(transformed).enhance(contrast)
        metadata = {
            "operation": operation,
            "brightness": round(brightness, 6),
            "contrast": round(contrast, 6),
        }
    elif operation == "gamma":
        spread = 0.10 if policy == "normal" else 0.20
        gamma = rng.uniform(1.0 - spread, 1.0 + spread)
        transformed = _gamma(grayscale, gamma)
        metadata = {"operation": operation, "gamma": round(gamma, 6)}
    elif operation == "noise":
        sigma = rng.uniform(1.0, 4.0) if policy == "normal" else rng.uniform(2.0, 8.0)
        transformed = _noise(grayscale, rng, sigma)
        metadata = {"operation": operation, "sigma": round(sigma, 6)}
    elif operation == "blur":
        radius = rng.uniform(0.2, 0.6) if policy == "normal" else rng.uniform(0.2, 1.0)
        transformed = grayscale.filter(ImageFilter.GaussianBlur(radius=radius))
        metadata = {"operation": operation, "radius": round(radius, 6)}
    else:
        clip_factor = rng.uniform(1.5, 3.0)
        transformed = _clahe_like(grayscale, clip_factor=clip_factor)
        metadata = {"operation": operation, "clip_factor": round(clip_factor, 6)}
    return transformed, metadata


def expanded_intersects(
    box: Box,
    crop: tuple[int, int, int, int],
    margin: int,
) -> bool:
    left, top, right, bottom = crop
    expanded_left = box.xmin - margin
    expanded_top = box.ymin - margin
    expanded_right = box.xmax + margin
    expanded_bottom = box.ymax + margin
    return min(right, expanded_right) > max(left, expanded_left) and min(
        bottom, expanded_bottom
    ) > max(top, expanded_top)


def find_safe_crop(
    image: Image.Image,
    boxes: Iterable[Box],
    rng: random.Random,
    crop_size: int,
    margin: int,
    attempts: int,
    min_std: float,
) -> tuple[int, int, int, int] | None:
    if crop_size <= 0 or margin < 0 or attempts <= 0 or min_std < 0:
        raise ValueError("crop parameters must be non-negative and attempts/crop_size positive")
    grayscale = image.convert("L")
    width, height = grayscale.size
    if width < crop_size or height < crop_size:
        return None
    box_list = tuple(boxes)
    for _ in range(attempts):
        left = rng.randint(0, width - crop_size)
        top = rng.randint(0, height - crop_size)
        crop = (left, top, left + crop_size, top + crop_size)
        if any(expanded_intersects(box, crop, margin) for box in box_list):
            continue
        patch = np.asarray(grayscale.crop(crop), dtype=np.float32)
        if float(patch.std()) < min_std:
            continue
        return crop
    return None
