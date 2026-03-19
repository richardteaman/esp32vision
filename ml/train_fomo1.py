#!/usr/bin/env python3
"""Experiment 1: current best baseline training config."""

from __future__ import annotations

from pathlib import Path

from experiment_runner import ROOT, run_core_script

RUN_NAME = "baseline_rgb_hard"
PREPARED_DIR = ROOT / "prepared"
OUTPUT_DIR = ROOT / "outputs" / RUN_NAME

INPUT_SIZE = 96
GRID_SIZE = 12
COLOR_MODE = "rgb"
TARGET_MODE = "hard"

BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 1e-3
LOSS = "focal"
VAL_SPLIT = 0.2
SEED = 42

# These are ignored for hard targets but kept here for consistency with the core script.
TARGET_SIGMA_CELLS = 0.8

NOISE_STD = 0.03
BRIGHTNESS_DELTA = 0.12
CONTRAST_LOWER = 0.85
CONTRAST_UPPER = 1.15


def main() -> None:
    run_core_script(
        "train_fomo.py",
        {
            "prepared_dir": PREPARED_DIR,
            "output_dir": OUTPUT_DIR,
            "input_size": INPUT_SIZE,
            "grid_size": GRID_SIZE,
            "color_mode": COLOR_MODE,
            "target_mode": TARGET_MODE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "loss": LOSS,
            "val_split": VAL_SPLIT,
            "seed": SEED,
            "target_sigma_cells": TARGET_SIGMA_CELLS,
            "noise_std": NOISE_STD,
            "brightness_delta": BRIGHTNESS_DELTA,
            "contrast_lower": CONTRAST_LOWER,
            "contrast_upper": CONTRAST_UPPER,
        },
    )


if __name__ == "__main__":
    main()
