#!/usr/bin/env python3
"""Prepare Edge Impulse bounding-box export for local FOMO-style training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Edge Impulse export into local metadata for FOMO-like training.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("../esp32-cam-coin_detection-export"),
        help="Path to exported Edge Impulse dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./prepared"),
        help="Where to write prepared manifests and previews.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=12,
        help="How many annotated preview images to render per split.",
    )
    return parser.parse_args()


def load_split(split_dir: Path) -> list[dict[str, Any]]:
    info_path = split_dir / "info.labels"
    with info_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["files"]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def resolve_image_path(split_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    options = []
    if candidate.is_absolute():
        options.append(candidate)
    else:
        options.append(split_dir / candidate)
        options.append(split_dir.parent / candidate)

    for option in options:
        if option.exists():
            return option

    raise FileNotFoundError(f"Could not resolve image path '{raw_path}' from {split_dir}")


def build_record(split_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    image_path = resolve_image_path(split_dir, item["path"])
    with Image.open(image_path) as image:
        width, height = image.size

    objects = []
    for bbox in item.get("boundingBoxes", []):
        center_x = bbox["x"] + bbox["width"] / 2.0
        center_y = bbox["y"] + bbox["height"] / 2.0
        objects.append(
            {
                "label": bbox["label"],
                "x": bbox["x"],
                "y": bbox["y"],
                "width": bbox["width"],
                "height": bbox["height"],
                "center_x": center_x,
                "center_y": center_y,
                "center_x_norm": clamp(center_x / width, 0.0, 0.999999),
                "center_y_norm": clamp(center_y / height, 0.0, 0.999999),
                "width_norm": clamp(bbox["width"] / width, 0.0, 1.0),
                "height_norm": clamp(bbox["height"] / height, 0.0, 1.0),
            }
        )

    return {
        "name": item["name"],
        "image_path": str(image_path.resolve()),
        "split": item["category"],
        "width": width,
        "height": height,
        "num_objects": len(objects),
        "objects": objects,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    box_hist = Counter(record["num_objects"] for record in records)
    labels = Counter(
        obj["label"]
        for record in records
        for obj in record["objects"]
    )
    non_empty = [record for record in records if record["num_objects"] > 0]
    total_boxes = sum(record["num_objects"] for record in records)
    avg_boxes = total_boxes / len(non_empty) if non_empty else 0.0
    return {
        "images": len(records),
        "non_empty_images": len(non_empty),
        "empty_images": len(records) - len(non_empty),
        "total_boxes": total_boxes,
        "avg_boxes_non_empty": round(avg_boxes, 3),
        "box_histogram": dict(sorted(box_hist.items())),
        "labels": dict(sorted(labels.items())),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def render_previews(
    records: list[dict[str, Any]],
    output_dir: Path,
    preview_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records[:preview_count]:
        with Image.open(record["image_path"]) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for obj in record["objects"]:
            x0 = obj["x"]
            y0 = obj["y"]
            x1 = x0 + obj["width"]
            y1 = y0 + obj["height"]
            cx = obj["center_x"]
            cy = obj["center_y"]
            draw.rectangle((x0, y0, x1, y1), outline=(255, 80, 80), width=2)
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(80, 255, 80))
        out_path = output_dir / f"{record['name']}.jpg"
        canvas.save(out_path, quality=90)


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    output_dir = args.output.resolve()

    split_dirs = {
        "train": source_dir / "training",
        "test": source_dir / "testing",
    }
    summary: dict[str, Any] = {
        "source": str(source_dir),
        "splits": {},
    }

    for split_name, split_dir in split_dirs.items():
        items = load_split(split_dir)
        records = [build_record(split_dir, item) for item in items]
        split_summary = summarize(records)
        summary["splits"][split_name] = split_summary
        save_json(output_dir / f"{split_name}.json", records)
        render_previews(
            records,
            output_dir / "previews" / split_name,
            args.preview_count,
        )

    save_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Prepared dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
