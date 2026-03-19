#!/usr/bin/env python3
"""Evaluate a trained FOMO-like model on the held-out test split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained FOMO-like model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, default=Path("./prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/eval"))
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--color-mode", choices=("grayscale", "rgb"), default="grayscale")
    parser.add_argument(
        "--decode-mode",
        choices=("components", "peaks"),
        default="components",
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--match-radius-cells", type=int, default=1)
    parser.add_argument("--peak-window", type=int, default=1)
    parser.add_argument("--peak-min-distance-cells", type=int, default=1)
    parser.add_argument("--preview-count", type=int, default=12)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_image(path: str, input_size: int, color_mode: str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L" if color_mode == "grayscale" else "RGB")
        image = image.resize((input_size, input_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
    if color_mode == "grayscale":
        return arr[..., np.newaxis]
    return arr


def grid_points_from_record(record: dict, grid_size: int) -> set[tuple[int, int]]:
    points: set[tuple[int, int]] = set()
    for obj in record["objects"]:
        gx = min(int(obj["center_x_norm"] * grid_size), grid_size - 1)
        gy = min(int(obj["center_y_norm"] * grid_size), grid_size - 1)
        points.add((gx, gy))
    return points


def centers_from_points(points: set[tuple[int, int]], width: int, height: int, grid_size: int):
    centers = []
    for gx, gy in sorted(points):
        cx = (gx + 0.5) * width / grid_size
        cy = (gy + 0.5) * height / grid_size
        centers.append((cx, cy))
    return centers


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []

    for gy in range(height):
        for gx in range(width):
            if not mask[gy, gx] or (gx, gy) in seen:
                continue
            stack = [(gx, gy)]
            seen.add((gx, gy))
            points: list[tuple[int, int]] = []

            while stack:
                cx, cy = stack.pop()
                points.append((cx, cy))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and (nx, ny) not in seen:
                            seen.add((nx, ny))
                            stack.append((nx, ny))

            components.append(points)

    return components


def heatmap_to_points(pred_map: np.ndarray, threshold: float) -> set[tuple[int, int]]:
    mask = pred_map >= threshold
    points: set[tuple[int, int]] = set()
    for component in connected_components(mask):
        best = max(component, key=lambda point: float(pred_map[point[1], point[0]]))
        points.add(best)
    return points


def heatmap_to_peaks(
    pred_map: np.ndarray,
    threshold: float,
    peak_window: int,
    peak_min_distance_cells: int,
) -> set[tuple[int, int]]:
    height, width = pred_map.shape
    candidates: list[tuple[float, int, int]] = []

    for gy in range(height):
        for gx in range(width):
            score = float(pred_map[gy, gx])
            if score < threshold:
                continue
            y0 = max(0, gy - peak_window)
            y1 = min(height, gy + peak_window + 1)
            x0 = max(0, gx - peak_window)
            x1 = min(width, gx + peak_window + 1)
            patch = pred_map[y0:y1, x0:x1]
            if score >= float(np.max(patch)):
                candidates.append((score, gx, gy))

    candidates.sort(reverse=True)
    points: list[tuple[int, int]] = []
    for _, gx, gy in candidates:
        too_close = False
        for px, py in points:
            if max(abs(px - gx), abs(py - gy)) <= peak_min_distance_cells:
                too_close = True
                break
        if not too_close:
            points.append((gx, gy))

    return set(points)


def match_points(
    gt_points: set[tuple[int, int]],
    pred_points: set[tuple[int, int]],
    radius_cells: int,
) -> tuple[int, int, int]:
    remaining_preds = set(pred_points)
    tp = 0

    for gx, gy in sorted(gt_points):
        best_pred = None
        best_distance = None
        for px, py in remaining_preds:
            distance = abs(px - gx) + abs(py - gy)
            if distance > radius_cells:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_pred = (px, py)
        if best_pred is not None:
            remaining_preds.remove(best_pred)
            tp += 1

    fp = len(remaining_preds)
    fn = len(gt_points) - tp
    return tp, fp, fn


def render_preview(
    record: dict,
    pred_points: set[tuple[int, int]],
    gt_points: set[tuple[int, int]],
    output_path: Path,
    grid_size: int,
) -> None:
    with Image.open(record["image_path"]) as image:
        canvas = image.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    for obj in record["objects"]:
        x0 = obj["x"]
        y0 = obj["y"]
        x1 = x0 + obj["width"]
        y1 = y0 + obj["height"]
        draw.rectangle((x0, y0, x1, y1), outline=(255, 80, 80), width=2)

    for cx, cy in centers_from_points(gt_points, width, height, grid_size):
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(80, 255, 80))

    for cx, cy in centers_from_points(pred_points, width, height, grid_size):
        draw.rectangle((cx - 4, cy - 4, cx + 4, cy + 4), outline=(80, 160, 255), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_records = load_records(args.prepared_dir / "test.json")
    x_test = np.stack(
        [
            load_image(record["image_path"], args.input_size, args.color_mode)
            for record in test_records
        ],
        axis=0,
    )

    import tensorflow as tf

    model = tf.keras.models.load_model(args.model)
    predictions = model.predict(x_test, verbose=0)

    tp = 0
    fp = 0
    fn = 0
    preview_items = []
    per_image = []

    for record, pred_map in zip(test_records, predictions):
        gt_points = grid_points_from_record(record, args.grid_size)
        if args.decode_mode == "peaks":
            pred_points = heatmap_to_peaks(
                pred_map[..., 0],
                args.threshold,
                args.peak_window,
                args.peak_min_distance_cells,
            )
        else:
            pred_points = heatmap_to_points(pred_map[..., 0], args.threshold)

        image_tp, image_fp, image_fn = match_points(
            gt_points,
            pred_points,
            args.match_radius_cells,
        )
        tp += image_tp
        fp += image_fp
        fn += image_fn

        mean_score = float(np.mean(pred_map))
        max_score = float(np.max(pred_map))
        item = {
            "name": record["name"],
            "gt_points": len(gt_points),
            "pred_points": len(pred_points),
            "tp": image_tp,
            "fp": image_fp,
            "fn": image_fn,
            "mean_score": mean_score,
            "max_score": max_score,
        }
        per_image.append(item)
        if len(preview_items) < args.preview_count:
            preview_items.append((record, pred_points, gt_points))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    summary = {
        "model": str(args.model.resolve()),
        "threshold": args.threshold,
        "match_radius_cells": args.match_radius_cells,
        "decode_mode": args.decode_mode,
        "peak_window": args.peak_window,
        "peak_min_distance_cells": args.peak_min_distance_cells,
        "input_size": args.input_size,
        "grid_size": args.grid_size,
        "color_mode": args.color_mode,
        "images": len(test_records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_image": per_image}, handle, ensure_ascii=True, indent=2)

    for record, pred_points, gt_points in preview_items:
        render_preview(
            record,
            pred_points,
            gt_points,
            args.output_dir / "previews" / f"{record['name']}.jpg",
            args.grid_size,
        )

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
