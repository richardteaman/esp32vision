#!/usr/bin/env python3
"""Export a trained Keras model to TFLite with optional int8 quantization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Keras model to TFLite.")
    parser.add_argument("--model", type=Path, required=True, help="Path to .keras model.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("./prepared"))
    parser.add_argument("--output", type=Path, default=Path("./outputs/model.tflite"))
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--color-mode", choices=("grayscale", "rgb"), default="grayscale")
    parser.add_argument(
        "--quantization",
        choices=("none", "int8"),
        default="int8",
    )
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=100,
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def representative_dataset(records: list[dict], input_size: int, limit: int, color_mode: str):
    for record in records[:limit]:
        with Image.open(record["image_path"]) as image:
            image = image.convert("L" if color_mode == "grayscale" else "RGB")
            image = image.resize((input_size, input_size))
            arr = np.asarray(image, dtype=np.float32) / 255.0
        if color_mode == "grayscale":
            arr = arr[..., np.newaxis]
        yield [arr[np.newaxis, ...]]


def main() -> None:
    args = parse_args()

    import tensorflow as tf

    model = tf.keras.models.load_model(args.model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantization == "int8":
        records = load_records(args.prepared_dir / "train.json")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(
            records, args.input_size, args.representative_samples, args.color_mode
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(tflite_model)
    print(f"Wrote {len(tflite_model)} bytes to {args.output.resolve()}")


if __name__ == "__main__":
    main()
