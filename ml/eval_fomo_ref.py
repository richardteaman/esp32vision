#!/usr/bin/env python3
"""Reference baseline evaluation run."""

from __future__ import annotations

from experiment_runner import ROOT, run_core_script

RUN_NAME = "baseline_rgb_hard_ref"
MODEL_PATH = ROOT / "outputs" / RUN_NAME / "best.keras"
PREPARED_DIR = ROOT / "prepared"
OUTPUT_DIR = ROOT / "outputs" / f"{RUN_NAME}_eval_peaks"

INPUT_SIZE = 96
GRID_SIZE = 12
COLOR_MODE = "rgb"

DECODE_MODE = "peaks"
THRESHOLD = 0.30
MATCH_RADIUS_CELLS = 1
PEAK_WINDOW = 1
PEAK_MIN_DISTANCE_CELLS = 1
PREVIEW_COUNT = 8


def main() -> None:
    run_core_script(
        "eval_fomo.py",
        {
            "model": MODEL_PATH,
            "prepared_dir": PREPARED_DIR,
            "output_dir": OUTPUT_DIR,
            "input_size": INPUT_SIZE,
            "grid_size": GRID_SIZE,
            "color_mode": COLOR_MODE,
            "decode_mode": DECODE_MODE,
            "threshold": THRESHOLD,
            "match_radius_cells": MATCH_RADIUS_CELLS,
            "peak_window": PEAK_WINDOW,
            "peak_min_distance_cells": PEAK_MIN_DISTANCE_CELLS,
            "preview_count": PREVIEW_COUNT,
        },
    )


if __name__ == "__main__":
    main()
