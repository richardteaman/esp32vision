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

### 5. Эксперимент На Устойчивость К Освещению

Если модель на реальной камере начинает недосчитывать монеты при других условиях света,
лучше не генерировать один фиксированный "аугментированный датасет" на диск, а усилить
on-the-fly аугментации во время обучения. Тогда на каждом epoch модель видит новые варианты
того же кадра.

Для отдельного lighting-robust прогона:

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision/ml
python3 train_fomo_lighting_ref.py
python3 eval_fomo_lighting_ref.py
```

Что усиливается в этом эксперименте:

- exposure / общая яркость
- gamma / нелинейное затемнение и пересвет
- contrast
- saturation и hue
- per-channel color scaling
- random shadow mask
- gaussian noise

Что смотреть после обучения:

- `ml/outputs/baseline_rgb_hard_light_aug/run_summary.json`
- `ml/outputs/baseline_rgb_hard_light_aug_eval_peaks/summary.json`
- `ml/outputs/baseline_rgb_hard_light_aug_eval_peaks/previews/...`

В `run_summary.json` теперь дополнительно сохраняются:

- `model_meta.total_params`
- `model_meta.trainable_params`
- `model_meta.non_trainable_params`

Если новый прогон выигрывает по качеству на реальных кадрах, дальше можно собрать новую
`int8` модель и обновить firmware bundle:

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision/ml
python3 export_tflite_lighting_ref.py
python3 eval_tflite_lighting_ref.py
python3 export_firmware_bundle_lighting_ref.py
```

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

## ДЗ 1-4

### ДЗ 1. Выбор проекта и датасет

Что сделано:

- выбран проект детекции монет на `ESP32-CAM`
- основной датасет хранится прямо в репозитории в `esp32-cam-coin_detection-export/`
- датасет уже разбит на `training/` и `testing/`
- этот датасет используется как единый источник для локальной подготовки выборки, обучения, оценки и прошивки

Как проверить:

```bash
cd /home/richt/Documents/PlatformIO/Projects/esp32vision
find esp32-cam-coin_detection-export -type f | wc -l
du -sh esp32-cam-coin_detection-export
```

Что должно подтвердиться:

- датасет лежит прямо в этой репе
- отдельная репа под датасет не нужна
- в `esp32-cam-coin_detection-export/` есть `training/` и `testing/`
- размер датасета небольшой и подходит для хранения вместе с кодом

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

То есть следующий этап оптимизации бдуeт связан уже не с датасетом, а с runtime/backend для инференса на ESP32.

## Update: Lighting-Robust Retrain

После дополнительных тестов на реальной камере в разных условиях освещения выяснилось,
что прежняя версия модели всё ещё могла заметно недосчитывать монеты: в сложных сценах
с изменённой яркостью и цветовым тоном модель иногда видела `2-3` монеты вместо `5-6`.

Что было изменено:

- добавлен отдельный retrain с усиленными on-the-fly photometric аугментациями:
  `exposure`, `gamma`, `brightness`, `contrast`, `saturation`, `hue`,
  `per-channel color scaling`, `random shadow mask`, `gaussian noise`
- сохранён rollback-резерв предыдущего firmware bundle:
  `include/coin_model_config_before_lighting_aug.h`
  `include/coin_model_data_before_lighting_aug.h`
- для новой модели firmware postprocessing выровнен с offline-eval:
  `threshold = 0.30`, `peak_min_distance_cells = 1`

Новые метрики новой lighting-robust модели:

- `Keras / held-out test`:
  `precision 0.9468`, `recall 0.8387`, `F1 0.8895`
- `int8 TFLite / held-out test`:
  `precision 0.9391`, `recall 0.8412`, `F1 0.8874`
- размер deploy-модели:
  `59840 bytes` (`~58.4 KB`)
- число параметров:
  `47617 total`, `47281 trainable`, `336 non-trainable`

Для сравнения, прежний baseline давал примерно:

- `Keras / held-out test`: `precision 0.941`, `recall 0.831`, `F1 0.883`
- `int8 TFLite / held-out test`: `precision 0.941`, `recall 0.834`, `F1 0.884`

Почему новый retrain помог:

- train-time distribution стала ближе к реальным кадрам с ESP32-CAM
- модель стала лучше переносить перепады яркости, локальные тени и цветовой сдвиг
- postprocessing на устройстве перестал быть жёстче, чем offline-eval

Итог по живым тестам:

- после lighting-robust retrain модель стала заметно стабильнее считать монеты
  при разных условиях освещения
- обновлённая версия уже записана в `include/coin_model_config.h`
  и `include/coin_model_data.h`
- при необходимости можно быстро откатиться на предыдущую версию через backup-файлы
