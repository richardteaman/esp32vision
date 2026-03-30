#!/usr/bin/env python3
"""Lighting-robust TFLite export run."""

from __future__ import annotations

from experiment_runner import ROOT, run_core_script

RUN_NAME = "baseline_rgb_hard_light_aug"
MODEL_PATH = ROOT / "outputs" / RUN_NAME / "best.keras"
PREPARED_DIR = ROOT / "prepared"
OUTPUT_PATH = ROOT / "outputs" / RUN_NAME / "model_int8.tflite"

INPUT_SIZE = 96
COLOR_MODE = "rgb"
QUANTIZATION = "int8"
REPRESENTATIVE_SAMPLES = 100


def main() -> None:
    run_core_script(
        "export_tflite.py",
        {
            "model": MODEL_PATH,
            "prepared_dir": PREPARED_DIR,
            "output": OUTPUT_PATH,
            "input_size": INPUT_SIZE,
            "color_mode": COLOR_MODE,
            "quantization": QUANTIZATION,
            "representative_samples": REPRESENTATIVE_SAMPLES,
        },
    )


if __name__ == "__main__":
    main()
