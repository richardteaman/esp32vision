#!/usr/bin/env python3
"""Train a small FOMO-like center detector locally."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight center-based detector for coins.",
    )
    parser.add_argument("--prepared-dir", type=Path, default=Path("./prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--color-mode", choices=("grayscale", "rgb"), default="grayscale")
    parser.add_argument("--target-mode", choices=("hard", "soft"), default="soft")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("focal", "bce"), default="focal")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sigma-cells", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--brightness-delta", type=float, default=0.12)
    parser.add_argument("--contrast-lower", type=float, default=0.85)
    parser.add_argument("--contrast-upper", type=float, default=1.15)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_train_val(
    records: list[dict],
    val_split: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_split))
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    if not train_records:
        raise ValueError("Validation split is too large for the dataset size.")
    return train_records, val_records


def load_image(path: str, input_size: int, color_mode: str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L" if color_mode == "grayscale" else "RGB")
        image = image.resize((input_size, input_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
    if color_mode == "grayscale":
        return arr[..., np.newaxis]
    return arr


def records_to_arrays(
    records: list[dict],
    input_size: int,
    grid_size: int,
    color_mode: str,
    target_mode: str,
    target_sigma_cells: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    channels = 1 if color_mode == "grayscale" else 3
    x_data = np.zeros((len(records), input_size, input_size, channels), dtype=np.float32)
    y_data = np.zeros((len(records), grid_size, grid_size, 1), dtype=np.float32)
    collisions = 0
    yy, xx = np.meshgrid(
        np.arange(grid_size, dtype=np.float32),
        np.arange(grid_size, dtype=np.float32),
        indexing="ij",
    )

    for idx, record in enumerate(records):
        x_data[idx] = load_image(record["image_path"], input_size, color_mode)
        target = np.zeros((grid_size, grid_size), dtype=np.float32)
        for obj in record["objects"]:
            gx = min(int(obj["center_x_norm"] * grid_size), grid_size - 1)
            gy = min(int(obj["center_y_norm"] * grid_size), grid_size - 1)
            if target[gy, gx] > 0.0:
                collisions += 1
            if target_mode == "hard":
                target[gy, gx] = 1.0
            else:
                center_x = min(obj["center_x_norm"] * grid_size, grid_size - 1e-4)
                center_y = min(obj["center_y_norm"] * grid_size, grid_size - 1e-4)
                dist_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
                heat = np.exp(-dist_sq / (2.0 * (target_sigma_cells ** 2)))
                target = np.maximum(target, heat)
        y_data[idx, ..., 0] = target

    meta = {
        "samples": len(records),
        "collisions": collisions,
        "collision_rate": collisions / len(records) if records else 0.0,
    }
    return x_data, y_data, meta


def build_model(input_size: int, grid_size: int, input_channels: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_size, input_size, input_channels))
    x = inputs
    for filters in (16, 24, 32):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    skip = x
    x = tf.keras.layers.Conv2D(48, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(48, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    skip = tf.keras.layers.Conv2D(48, 1, padding="same", use_bias=False)(skip)
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SpatialDropout2D(0.1)(x)
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if model.output_shape[1] != grid_size or model.output_shape[2] != grid_size:
        raise ValueError(
            "Grid size mismatch. "
            f"Model output is {model.output_shape[1:3]}, expected {(grid_size, grid_size)}."
        )
    return model


def build_loss(loss_name: str):
    import tensorflow as tf

    if loss_name == "focal":
        return tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
    return tf.keras.losses.BinaryCrossentropy()


def make_train_dataset(x_data: np.ndarray, y_data: np.ndarray, args: argparse.Namespace):
    import tensorflow as tf

    def augment(image, target):
        flip_lr = tf.random.uniform(()) > 0.5
        flip_ud = tf.random.uniform(()) > 0.5
        image = tf.cond(flip_lr, lambda: tf.image.flip_left_right(image), lambda: image)
        target = tf.cond(flip_lr, lambda: tf.image.flip_left_right(target), lambda: target)
        image = tf.cond(flip_ud, lambda: tf.image.flip_up_down(image), lambda: image)
        target = tf.cond(flip_ud, lambda: tf.image.flip_up_down(target), lambda: target)
        image = tf.image.random_brightness(image, max_delta=args.brightness_delta)
        image = tf.image.random_contrast(
            image,
            lower=args.contrast_lower,
            upper=args.contrast_upper,
        )
        noise = tf.random.normal(tf.shape(image), stddev=args.noise_std)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)
        return image, target

    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(len(x_data), seed=args.seed, reshuffle_each_iteration=True)
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    all_train_records = load_records(args.prepared_dir / "train.json")
    train_records, val_records = split_train_val(
        all_train_records, args.val_split, args.seed
    )
    test_records = load_records(args.prepared_dir / "test.json")
    if args.max_train_samples > 0:
        train_records = train_records[: args.max_train_samples]
        val_records = val_records[: max(1, args.max_train_samples // 4)]
    if args.max_test_samples > 0:
        test_records = test_records[: args.max_test_samples]
        val_records = val_records[: min(len(val_records), args.max_test_samples)]

    x_train, y_train, train_meta = records_to_arrays(
        train_records,
        args.input_size,
        args.grid_size,
        args.color_mode,
        args.target_mode,
        args.target_sigma_cells,
    )
    x_val, y_val, val_meta = records_to_arrays(
        val_records,
        args.input_size,
        args.grid_size,
        args.color_mode,
        args.target_mode,
        args.target_sigma_cells,
    )

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)

    channels = 1 if args.color_mode == "grayscale" else 3
    model = build_model(args.input_size, args.grid_size, channels)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=build_loss(args.loss),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
    ]

    train_dataset = make_train_dataset(x_train, y_train, args)

    history = model.fit(
        train_dataset,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    eval_metrics = model.evaluate(x_val, y_val, verbose=0, return_dict=True)
    model.save(args.output_dir / "final.keras")

    save_json(
        args.output_dir / "run_summary.json",
        {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "train_meta": train_meta,
            "val_meta": val_meta,
            "held_out_test_samples": len(test_records),
            "eval_metrics": eval_metrics,
            "history": history.history,
        },
    )
    print(json.dumps(eval_metrics, ensure_ascii=True, indent=2))
    print(f"Saved model and metrics to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
