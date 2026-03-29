#!/usr/bin/env python3
"""Diagnose real ESP32-CAM captures against the local int8 model.

Run this with the ML virtualenv so TensorFlow and Pillow are available:
  ../../ml/.venv/bin/python diagnose_real_capture.py --base-url http://192.168.1.123
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is not installed in the current interpreter. "
        "Run this script with ml/.venv/bin/python."
    ) from exc

from eval_fomo import heatmap_to_peaks
from eval_tflite import dequantize_output, quantize_input


@dataclass(frozen=True)
class Variant:
    name: str
    preprocess: str
    threshold: float
    peak_min_distance_cells: int


VARIANTS = (
    Variant("stretch_rgb_ref", "stretch_rgb", 0.30, 1),
    Variant("stretch_rgb_fw", "stretch_rgb", 0.35, 2),
    Variant("stretch_bgr_ref", "stretch_bgr", 0.30, 1),
    Variant("center_crop_rgb_ref", "center_crop_rgb", 0.30, 1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose real ESP32-CAM captures with several preprocessing variants.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--base-url", help="ESP32-CAM base URL, for example http://192.168.1.123")
    source_group.add_argument("--image", type=Path, help="Local JPEG or PNG file to analyze")
    parser.add_argument("--frames", type=int, default=8, help="How many frames to fetch from /capture")
    parser.add_argument("--delay", type=float, default=0.4, help="Delay between frame fetches in seconds")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds")
    parser.add_argument("--capture-endpoint", default="/capture", help="Capture endpoint path")
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "ml" / "outputs" / "baseline_rgb_hard_ref" / "model_int8.tflite",
        help="Path to the int8 TFLite model",
    )
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "diagnostics" / time.strftime("%Y%m%d_%H%M%S"),
        help="Where to write captured frames, overlays and summary JSON",
    )
    return parser.parse_args()


def fetch_frame(base_url: str, capture_endpoint: str, timeout: float) -> bytes:
    url = base_url.rstrip("/") + capture_endpoint
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_pil_image(frame_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(frame_bytes)) as image:
        return image.convert("RGB")


def preprocess_image(image: Image.Image, mode: str, input_size: int) -> np.ndarray:
    canvas = image.convert("RGB")
    if mode == "center_crop_rgb":
        width, height = canvas.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        canvas = canvas.crop((left, top, left + side, top + side))

    canvas = canvas.resize((input_size, input_size), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    if mode == "stretch_bgr":
        arr = arr[..., ::-1]
    return arr


def run_inference(
    interpreter: tf.lite.Interpreter,
    input_detail: dict,
    output_detail: dict,
    image_arr: np.ndarray,
) -> np.ndarray:
    input_tensor = image_arr[np.newaxis, ...]
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


def centers_from_points(points: set[tuple[int, int]], width: int, height: int, grid_size: int) -> list[tuple[float, float]]:
    centers = []
    for gx, gy in sorted(points):
        cx = (gx + 0.5) * width / grid_size
        cy = (gy + 0.5) * height / grid_size
        centers.append((cx, cy))
    return centers


def render_overlay(
    image: Image.Image,
    pred_points: set[tuple[int, int]],
    grid_size: int,
    variant: Variant,
    max_score: float,
    output_path: Path,
) -> None:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    for cx, cy in centers_from_points(pred_points, width, height, grid_size):
        draw.rectangle((cx - 5, cy - 5, cx + 5, cy + 5), outline=(80, 160, 255), width=2)

    draw.text((8, 8), f"{variant.name} count={len(pred_points)} max={max_score:.3f}", fill=(255, 255, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def summarize_variant(items: list[dict]) -> dict:
    counts = [item["count"] for item in items]
    max_scores = [item["max_score"] for item in items]
    histogram: dict[str, int] = {}
    for value in counts:
        key = str(value)
        histogram[key] = histogram.get(key, 0) + 1

    return {
        "frames": len(items),
        "mean_count": statistics.fmean(counts) if counts else 0.0,
        "count_stdev": statistics.pstdev(counts) if len(counts) > 1 else 0.0,
        "count_histogram": histogram,
        "mean_max_score": statistics.fmean(max_scores) if max_scores else 0.0,
        "max_score_max": max(max_scores) if max_scores else 0.0,
        "max_score_min": min(max_scores) if max_scores else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    frame_payloads: list[tuple[str, Image.Image]] = []
    if args.image is not None:
        frame_bytes = args.image.read_bytes()
        frame_payloads.append((args.image.stem, load_pil_image(frame_bytes)))
    else:
        for index in range(args.frames):
            frame_name = f"frame_{index:03d}"
            frame_bytes = fetch_frame(args.base_url, args.capture_endpoint, args.timeout)
            frame_payloads.append((frame_name, load_pil_image(frame_bytes)))
            (args.output_dir / "frames").mkdir(parents=True, exist_ok=True)
            (args.output_dir / "frames" / f"{frame_name}.jpg").write_bytes(frame_bytes)
            if index + 1 < args.frames:
                time.sleep(args.delay)

    per_frame: list[dict] = []
    variant_groups: dict[str, list[dict]] = {variant.name: [] for variant in VARIANTS}

    for frame_name, image in frame_payloads:
        frame_record = {
            "frame": frame_name,
            "variants": [],
        }

        for variant in VARIANTS:
            input_arr = preprocess_image(image, variant.preprocess, args.input_size)
            pred_map = run_inference(interpreter, input_detail, output_detail, input_arr)
            pred_points = heatmap_to_peaks(
                pred_map[..., 0],
                variant.threshold,
                peak_window=1,
                peak_min_distance_cells=variant.peak_min_distance_cells,
            )

            variant_record = {
                "name": variant.name,
                "preprocess": variant.preprocess,
                "threshold": variant.threshold,
                "peak_min_distance_cells": variant.peak_min_distance_cells,
                "count": len(pred_points),
                "max_score": float(np.max(pred_map)),
                "mean_score": float(np.mean(pred_map)),
                "points": sorted([{"gx": gx, "gy": gy} for gx, gy in pred_points], key=lambda item: (item["gy"], item["gx"])),
            }
            frame_record["variants"].append(variant_record)
            variant_groups[variant.name].append(variant_record)

            render_overlay(
                image,
                pred_points,
                args.grid_size,
                variant,
                variant_record["max_score"],
                args.output_dir / "overlays" / variant.name / f"{frame_name}.jpg",
            )

        per_frame.append(frame_record)

    summary = {
        "source": str(args.image.resolve()) if args.image is not None else args.base_url,
        "model": str(args.model.resolve()),
        "frames": len(frame_payloads),
        "input_dtype": str(np.dtype(input_detail["dtype"])),
        "output_dtype": str(np.dtype(output_detail["dtype"])),
        "input_quantization": list(input_detail["quantization"]),
        "output_quantization": list(output_detail["quantization"]),
        "variants": {name: summarize_variant(items) for name, items in variant_groups.items()},
    }

    payload = {
        "summary": summary,
        "per_frame": per_frame,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Saved diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
