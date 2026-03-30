#!/usr/bin/env python3
"""Lighting-robust training run with stronger photometric augmentation."""

from __future__ import annotations

from experiment_runner import ROOT, run_core_script

RUN_NAME = "baseline_rgb_hard_light_aug"
PREPARED_DIR = ROOT / "prepared"
OUTPUT_DIR = ROOT / "outputs" / RUN_NAME

INPUT_SIZE = 96
GRID_SIZE = 12
COLOR_MODE = "rgb"
TARGET_MODE = "hard"

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-3
LOSS = "focal"
VAL_SPLIT = 0.2
SEED = 42

TARGET_SIGMA_CELLS = 0.8

NOISE_STD = 0.04
BRIGHTNESS_DELTA = 0.20
CONTRAST_LOWER = 0.70
CONTRAST_UPPER = 1.35
EXPOSURE_LOWER = 0.60
EXPOSURE_UPPER = 1.45
GAMMA_LOWER = 0.70
GAMMA_UPPER = 1.50
SATURATION_LOWER = 0.65
SATURATION_UPPER = 1.40
HUE_DELTA = 0.04
CHANNEL_SCALE_MAX_DELTA = 0.18
SHADOW_PROB = 0.50
SHADOW_STRENGTH_MAX = 0.45


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
            "exposure_lower": EXPOSURE_LOWER,
            "exposure_upper": EXPOSURE_UPPER,
            "gamma_lower": GAMMA_LOWER,
            "gamma_upper": GAMMA_UPPER,
            "saturation_lower": SATURATION_LOWER,
            "saturation_upper": SATURATION_UPPER,
            "hue_delta": HUE_DELTA,
            "channel_scale_max_delta": CHANNEL_SCALE_MAX_DELTA,
            "shadow_prob": SHADOW_PROB,
            "shadow_strength_max": SHADOW_STRENGTH_MAX,
        },
    )


if __name__ == "__main__":
    main()
