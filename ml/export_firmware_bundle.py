#!/usr/bin/env python3
"""Export a TFLite model into firmware-friendly headers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TFLite model into firmware headers.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--data-output", type=Path, required=True)
    parser.add_argument("--detection-threshold", type=float, default=0.3)
    parser.add_argument("--peak-window", type=int, default=1)
    parser.add_argument("--peak-min-distance-cells", type=int, default=1)
    parser.add_argument("--match-radius-cells", type=int, default=1)
    parser.add_argument("--array-name", default="g_coin_model_data")
    parser.add_argument("--namespace", default="coin_model")
    return parser.parse_args()


def format_bytes(data: bytes, row_width: int = 12) -> str:
    rows = []
    for offset in range(0, len(data), row_width):
        chunk = data[offset : offset + row_width]
        rows.append("  " + ", ".join(f"0x{value:02x}" for value in chunk))
    return ",\n".join(rows)


def write_config_header(
    output_path: Path,
    namespace: str,
    model_path: Path,
    input_detail: dict,
    output_detail: dict,
    args: argparse.Namespace,
) -> None:
    input_shape = input_detail["shape"]
    output_shape = output_detail["shape"]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]

    text = f"""#pragma once

// Auto-generated from: {model_path.resolve()}

#include <cstddef>
#include <cstdint>

namespace {namespace} {{

constexpr int kInputHeight = {int(input_shape[1])};
constexpr int kInputWidth = {int(input_shape[2])};
constexpr int kInputChannels = {int(input_shape[3])};
constexpr std::size_t kInputTensorBytes = static_cast<std::size_t>(kInputHeight) * kInputWidth * kInputChannels;

constexpr int kOutputGridHeight = {int(output_shape[1])};
constexpr int kOutputGridWidth = {int(output_shape[2])};
constexpr int kOutputChannels = {int(output_shape[3])};
constexpr std::size_t kOutputTensorBytes = static_cast<std::size_t>(kOutputGridHeight) * kOutputGridWidth * kOutputChannels;

constexpr float kInputScale = {float(input_scale):.10f}f;
constexpr int kInputZeroPoint = {int(input_zero_point)};
constexpr float kOutputScale = {float(output_scale):.10f}f;
constexpr int kOutputZeroPoint = {int(output_zero_point)};

constexpr float kDetectionThreshold = {float(args.detection_threshold):.4f}f;
constexpr int kPeakWindow = {int(args.peak_window)};
constexpr int kPeakMinDistanceCells = {int(args.peak_min_distance_cells)};
constexpr int kMatchRadiusCells = {int(args.match_radius_cells)};

}}  // namespace {namespace}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def write_data_header(
    output_path: Path,
    namespace: str,
    array_name: str,
    model_path: Path,
    data: bytes,
) -> None:
    body = format_bytes(data)
    text = f"""#pragma once

// Auto-generated from: {model_path.resolve()}

#include <cstddef>
#include <cstdint>

namespace {namespace} {{

alignas(16) const unsigned char {array_name}[] = {{
{body}
}};

constexpr std::size_t {array_name}_len = {len(data)};

}}  // namespace {namespace}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise ValueError("Firmware export expects an int8 TFLite model.")

    model_bytes = args.model.read_bytes()

    write_config_header(
        args.config_output,
        args.namespace,
        args.model,
        input_detail,
        output_detail,
        args,
    )
    write_data_header(
        args.data_output,
        args.namespace,
        args.array_name,
        args.model,
        model_bytes,
    )

    print(f"Wrote config header to: {args.config_output.resolve()}")
    print(f"Wrote model header to: {args.data_output.resolve()}")


if __name__ == "__main__":
    main()
