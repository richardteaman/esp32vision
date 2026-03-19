#!/usr/bin/env python3
"""Evaluate a TFLite FOMO-like model on the held-out test split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from eval_fomo import (
    grid_points_from_record,
    heatmap_to_peaks,
    heatmap_to_points,
    load_image,
    load_records,
    match_points,
    render_preview,
)

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TFLite FOMO-like model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, default=Path("./prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/eval_tflite"))
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


def quantize_input(arr: np.ndarray, input_detail: dict) -> np.ndarray:
    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        return arr.astype(input_detail["dtype"])
    quantized = np.round(arr / scale + zero_point)
    dtype = np.dtype(input_detail["dtype"])
    quantized = np.clip(quantized, np.iinfo(dtype).min, np.iinfo(dtype).max)
    return quantized.astype(dtype)


def dequantize_output(arr: np.ndarray, output_detail: dict) -> np.ndarray:
    scale, zero_point = output_detail["quantization"]
    if scale == 0:
        return arr.astype(np.float32)
    return (arr.astype(np.float32) - zero_point) * scale


def run_inference(interpreter, input_detail: dict, output_detail: dict, image: np.ndarray) -> np.ndarray:
    input_tensor = image[np.newaxis, ...]
    if np.issubdtype(input_detail["dtype"], np.integer):
        input_tensor = quantize_input(input_tensor, input_detail)
    else:
        input_tensor = input_tensor.astype(input_detail["dtype"])

    interpreter.set_tensor(input_detail["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])[0]

    if np.issubdtype(output_detail["dtype"], np.integer):
        return dequantize_output(output, output_detail)
    return output.astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    test_records = load_records(args.prepared_dir / "test.json")

    tp = 0
    fp = 0
    fn = 0
    preview_items = []
    per_image = []

    for record in test_records:
        image = load_image(record["image_path"], args.input_size, args.color_mode)
        pred_map = run_inference(interpreter, input_detail, output_detail, image)
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

        item = {
            "name": record["name"],
            "gt_points": len(gt_points),
            "pred_points": len(pred_points),
            "tp": image_tp,
            "fp": image_fp,
            "fn": image_fn,
            "mean_score": float(np.mean(pred_map)),
            "max_score": float(np.max(pred_map)),
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
        "input_dtype": str(np.dtype(input_detail["dtype"])),
        "output_dtype": str(np.dtype(output_detail["dtype"])),
        "input_quantization": list(input_detail["quantization"]),
        "output_quantization": list(output_detail["quantization"]),
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
