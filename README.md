# ESP32 Coin Detection

Локальный проект по детекции монет на `ESP32-CAM`.
Разметка и исходный экспорт датасета взяты из `Edge Impulse`, но обучение, квантизация, экспорт модели и отладка пайплайна выполняются локально.

Сейчас в репозитории уже есть:

- локальный FOMO-style pipeline для обучения и оценки на ПК
- `int8`-экспорт модели для `TFLite Micro`
- прошивка для `ESP32-CAM` с HTTP-endpoints для захвата кадра, инференса и диагностики
- отдельный скрипт для сравнения реальных кадров с камеры и локального `TFLite`-инференса

## Структура

- `src/` - прошивка ESP32-CAM
- `include/` - заголовки модели и конфиг для firmware
- `ml/` - подготовка датасета, обучение, оценка, экспорт `TFLite`
- `tools/scripts/` - утилиты для работы с камерой и диагностики реальных кадров
- `esp32-cam-coin_detection-export/` - экспорт датасета из Edge Impulse
- `platformio.ini` - окружение `esp32cam` для PlatformIO

Подробности по ML-части отдельно описаны в [ml/README.md](ml/README.md).

## Что Нужно Для Запуска

- Python 3
- `venv`
- VS Code + PlatformIO extension
- плата `AI Thinker ESP32-CAM`

## Быстрый Старт

### 1. Поднять Python-окружение

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision/ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Подготовить датасет

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision/ml
python3 prepare_dataset.py
```

Что появится после запуска:

- `ml/prepared/train.json`
- `ml/prepared/test.json`
- `ml/prepared/summary.json`
- `ml/prepared/previews/...`

Ожидаемая сводка по текущему датасету:

- train: `490` изображений
- test: `123` изображений
- train empty: `39`
- test empty: `12`
- класс: `coin`

### 3. Обучить reference baseline на ПК

Самый удобный воспроизводимый запуск:

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision/ml
python3 train_fomo_ref.py
python3 eval_fomo_ref.py
```

Если нужен запуск с явными параметрами:

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision/ml
python3 train_fomo.py \
  --input-size 96 \
  --grid-size 12 \
  --color-mode rgb \
  --target-mode hard \
  --epochs 30 \
  --batch-size 16 \
  --loss focal
```

Ожидаемый baseline на held-out test:

- precision: `0.941`
- recall: `0.831`
- F1: `0.883`

Смотреть результат:

- `ml/outputs/baseline_rgb_hard_ref_eval_peaks/summary.json`
- `ml/outputs/baseline_rgb_hard_ref_eval_peaks/previews/...`

### 4. Сделать квантизацию и экспорт

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision/ml
python3 export_tflite_ref.py
python3 eval_tflite_ref.py
python3 export_firmware_bundle_ref.py
```

Ожидаемый результат после `int8`-квантизации:

- precision: `0.941`
- recall: `0.834`
- F1: `0.884`

Что проверить:

- `ml/outputs/baseline_rgb_hard_ref/model_int8.tflite`
- `ml/outputs/baseline_rgb_hard_ref_int8_eval_peaks/summary.json`
- `include/coin_model_config.h`
- `include/coin_model_data.h`

## Firmware И Проверка На ESP32-CAM

Часть со сборкой через PlatformIO предполагается ручной в VS Code.

Что сделать:

1. Открыть папку проекта в VS Code.
2. Убедиться, что выбрано окружение `esp32cam`.
3. Перед первой прошивкой поменять `ssid` и `password` в `src/main.cpp`.
4. Собрать и прошить проект через PlatformIO UI.
5. Открыть Serial Monitor и узнать IP камеры.

После запуска доступны endpoints:

- `GET /` - краткая справка по API
- `GET /capture` - одиночный JPEG-кадр
- `GET /stream` - MJPEG stream
- `GET /detect/preprocess` - статистика входного тензора
- `GET /detect/run` - основной рабочий путь: `JPEG/QVGA -> RGB888 -> resize -> infer`
- `GET /detect/run_rgb565` - старый direct `RGB565` path, оставлен для сравнения
- `GET /detect/debug/channels` - сравнение `RGB` и `BGR` на одном direct-frame

Для проверки на реальной камере:

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision
./ml/.venv/bin/python tools/scripts/diagnose_real_capture.py --base-url http://<ESP32_IP> --frames 10
```

Скрипт сохранит:

- `tools/scripts/diagnostics/<timestamp>/summary.json`
- `tools/scripts/diagnostics/<timestamp>/frames/...`
- `tools/scripts/diagnostics/<timestamp>/overlays/...`

## ДЗ 2-4

### ДЗ 2. Первичный датасет и первая модель

Что сделано:

- использован экспорт датасета из `esp32-cam-coin_detection-export/`
- датасет приведён к локальному формату через `ml/prepare_dataset.py`
- подготовлен первый локальный FOMO-style pipeline
- выбрана рабочая конфигурация baseline: `RGB`, `96x96`, `12x12`, `hard target`, `focal loss`

Как проверить:

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision/ml
python3 prepare_dataset.py
python3 train_fomo_ref.py
```

Что должно подтвердиться:

- в `ml/prepared/summary.json` есть train/test split
- после обучения появляется baseline model в `ml/outputs/...`
- модель обучается локально без Edge Impulse training pipeline

### ДЗ 3. Baseline на ПК

Что сделано:

- baseline обучен и оценён на held-out test
- добавлена отдельная офлайн-оценка по heatmap peaks
- сохранены summary и preview-артефакты для проверки качества

Как проверить:

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision/ml
python3 eval_fomo_ref.py
```

Что должно подтвердиться:

- `ml/outputs/baseline_rgb_hard_ref_eval_peaks/summary.json` содержит метрики около:
  `precision 0.941`, `recall 0.831`, `F1 0.883`
- previews показывают адекватные попадания по центрам монет

### ДЗ 4. Квантизация

Что сделано:

- baseline экспортирован в `int8 TFLite`
- отдельно проверено, что после квантизации качество почти не просело
- сгенерирован firmware bundle для `ESP32-CAM`

Как проверить:

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision/ml
python3 export_tflite_ref.py
python3 eval_tflite_ref.py
python3 export_firmware_bundle_ref.py
```

Что должно подтвердиться:

- появляется `model_int8.tflite`
- `ml/outputs/baseline_rgb_hard_ref_int8_eval_peaks/summary.json` содержит метрики около:
  `precision 0.941`, `recall 0.834`, `F1 0.884`
- `include/coin_model_config.h` и `include/coin_model_data.h` обновлены под firmware

## Что Уже Проверено На Реальном ESP32-CAM

Во время диагностики выяснилось:

- старая схема `RGB565 96x96 -> infer` работала плохо и галлюцинировала лишние монеты
- проблема была не в самой `int8`-модели, а в несовпадении firmware-preprocessing с train/eval pipeline
- основной endpoint `GET /detect/run` переведён на путь `JPEG/QVGA -> RGB888 -> resize -> quantize -> infer`
- после этого на реальных тестах были получены корректные счёты:
  одна монета -> `count = 1`
  две монеты -> `count = 2`
  две монеты -> `count = 2`

Проверка через `diagnose_real_capture.py` на `10` реальных кадрах показала:

- `stretch_rgb_ref`: `mean_count 1.2`
- `stretch_rgb_fw`: `mean_count 1.1`
- `stretch_bgr_ref`: `mean_count 0.9`
- `center_crop_rgb_ref`: `mean_count 1.0`

Это подтверждает, что текущий главный фикс был именно в preprocessing pipeline.

## Текущее Ограничение

По качеству проект стал заметно лучше, но по скорости проблема ещё остаётся:

- preprocess на устройстве занимает примерно `120 ms`
- сам `Invoke()` занимает примерно `15.6 s`

То есть следующий этап оптимизации бдут связан уже не с датасетом, а с runtime/backend для инференса на ESP32.
