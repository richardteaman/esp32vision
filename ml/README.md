# Local FOMO-style pipeline

We use Edge Impulse only for labeling. Training, quantization and deployment are local.

## Why FOMO-style detection

- One object class (`coin`) and small objects fit center-based detection well.
- ESP32-CAM is memory-constrained, so center heatmaps are simpler than full bounding-box regressors.
- The final MCU model can report coin centers or count coins without a heavy post-processing stage.

## Current dataset

- Source: `../esp32-cam-coin_detection-export`
- Train images: 490
- Test images: 123
- Labels: one class, `coin`

## Setup

Create a virtual environment in `ml` and install dependencies:

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want fixed experiment scripts without typing CLI flags, use:

```bash
cd ml
python3 train_fomo1.py
python3 eval_fomo1.py
```

The parameters for each experiment live at the top of the corresponding script.

## Step 1. Prepare dataset

```bash
cd ml
python3 prepare_dataset.py
```

This generates:

- `prepared/train.json`
- `prepared/test.json`
- `prepared/summary.json`
- `prepared/previews/...`

The prepared format stores:

- absolute path to the image
- image size
- raw bounding boxes
- normalized coin centers

## Step 2. Train baseline

```bash
cd ml
python3 train_fomo.py \
  --input-size 96 \
  --grid-size 12 \
  --color-mode rgb \
  --target-mode hard \
  --epochs 30 \
  --batch-size 16 \
  --loss focal
```

Quick smoke test:

```bash
cd ml
python3 train_fomo.py \
  --epochs 1 \
  --max-train-samples 32 \
  --max-test-samples 16 \
  --color-mode rgb \
  --target-mode hard \
  --loss focal
```

What this does:

- trains on `prepared/train.json`
- keeps part of training data as validation
- does not touch `prepared/test.json`

Notes:

- Best current baseline uses RGB `96x96`.
- Output is a `12x12x1` grid.
- A cell is positive if it contains a coin center.
- With the current dataset, `12x12` gives low target collisions: 10 on train and 1 on test.
- Default loss is `focal`, because positive cells are sparse and plain BCE tends to collapse to all-background predictions.
- Recommended baseline target mode is `hard`.
- Augmentations are enabled during training: horizontal/vertical flips, brightness, contrast and small Gaussian noise.
- The train script also supports stronger lighting robustness augmentation: exposure, gamma, saturation, hue, per-channel color scaling and random shadow masks.
- Augmentations are generated on the fly during training, so you do not need to pre-render a separate augmented dataset on disk.

That is a local analogue of the FOMO idea and is realistic for ESP32-class deployment.

## Step 3. Evaluate baseline on held-out test

```bash
cd ml
python3 eval_fomo.py \
  --model outputs/best.keras \
  --color-mode rgb \
  --threshold 0.35 \
  --match-radius-cells 1
```

This generates:

- `outputs/eval/summary.json`
- `outputs/eval/previews/...`

The evaluation:

- thresholds the output heatmap
- merges neighboring active cells into one prediction
- matches predicted centers to ground-truth centers with a small grid-cell tolerance

This is closer to how a FOMO-style detector should be interpreted than counting every active cell as a separate coin.

## Step 4. Export TFLite

```bash
cd ml
python3 export_tflite.py --model outputs/best.keras --quantization int8
```

This produces a quantized `.tflite` model for the next integration step.

Reference run without manual flags:

```bash
cd ml
python3 export_tflite_ref.py
python3 eval_tflite_ref.py
```

To export firmware-ready headers for the ESP32 project:

```bash
cd ml
python3 export_firmware_bundle_ref.py
```

## Next firmware step

After the baseline is trained and exported:

1. move ESP32-CAM capture from JPEG-only mode to an inference-friendly frame format
2. replicate resize and grayscale preprocessing on-device
3. run TFLite Micro inference
4. convert positive grid cells into coin centers
5. output centers or count via Serial/HTTP

The project now includes an intermediate deploy step in firmware:

- `GET /detect/preprocess`
- switches camera to `RGB565 96x96`
- converts the frame into the same `int8` input tensor format as the quantized model
- returns JSON stats for the prepared tensor

## Homework mapping

1. Primary dataset: already collected and labeled.
2. Baseline on PC:
   run `train_fomo.py`, then `eval_fomo.py`, save metrics and previews.
3. Quantization:
   after baseline, export `int8` and compare metrics before and after quantization.

## Best current baseline

- Train config:
  `rgb`, `96x96`, `12x12`, `hard` target, `focal loss`
- Held-out test result:
  `precision 0.941`, `recall 0.831`, `F1 0.883`

## Experiment launchers

- `train_fomo1.py` / `eval_fomo1.py`: current best baseline
- `train_fomo2.py` / `eval_fomo2.py`: denser-grid experiment for crowded scenes
- `train_fomo_ref.py` / `eval_fomo_ref.py`: fixed reference baseline
- `train_fomo_lighting_ref.py` / `eval_fomo_lighting_ref.py`: stronger lighting/color augmentation for camera robustness
- `export_tflite_ref.py` / `eval_tflite_ref.py`: fixed reference int8 export and eval
- `export_tflite_lighting_ref.py` / `eval_tflite_lighting_ref.py`: int8 export and eval for the lighting-robust run
- `export_firmware_bundle_ref.py`: generate `include/coin_model_config.h` and `include/coin_model_data.h`
- `export_firmware_bundle_lighting_ref.py`: generate firmware headers from the lighting-robust int8 model

## What to check after the first run

- false positives on empty frames
- missed coins when coins touch each other
- grid collisions when two coin centers fall into one output cell
- model size after int8 quantization
