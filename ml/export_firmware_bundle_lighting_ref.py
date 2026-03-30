#!/usr/bin/env python3
"""Lighting-robust firmware bundle export run."""

from __future__ import annotations

from experiment_runner import ROOT, run_core_script

RUN_NAME = "baseline_rgb_hard_light_aug"
MODEL_PATH = ROOT / "outputs" / RUN_NAME / "model_int8.tflite"
CONFIG_OUTPUT = ROOT.parent / "include" / "coin_model_config.h"
DATA_OUTPUT = ROOT.parent / "include" / "coin_model_data.h"

DETECTION_THRESHOLD = 0.30
PEAK_WINDOW = 1
PEAK_MIN_DISTANCE_CELLS = 1
MATCH_RADIUS_CELLS = 1


def main() -> None:
    run_core_script(
        "export_firmware_bundle.py",
        {
            "model": MODEL_PATH,
            "config_output": CONFIG_OUTPUT,
            "data_output": DATA_OUTPUT,
            "detection_threshold": DETECTION_THRESHOLD,
            "peak_window": PEAK_WINDOW,
            "peak_min_distance_cells": PEAK_MIN_DISTANCE_CELLS,
            "match_radius_cells": MATCH_RADIUS_CELLS,
        },
    )


if __name__ == "__main__":
    main()
