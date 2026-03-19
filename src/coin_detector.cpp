#include "coin_detector.h"

#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>

#include "coin_model_config.h"
#include "coin_model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr std::size_t kTensorArenaSize = 224 * 1024;

tflite::MicroErrorReporter g_micro_error_reporter;
tflite::ErrorReporter* g_error_reporter = &g_micro_error_reporter;
const tflite::Model* g_model = nullptr;
tflite::AllOpsResolver g_resolver;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;
uint8_t* g_tensor_arena = nullptr;

CoinDetectionResult g_last_result = {};
const char* g_last_error = "Detector not initialized";

void set_error(const char* message) {
  g_last_error = message;
  g_last_result.ok = false;
  g_last_result.error = message;
}

float dequantize_score(int8_t value) {
  return (static_cast<int>(value) - coin_model::kOutputZeroPoint) * coin_model::kOutputScale;
}

bool is_local_peak(const int8_t* output, int gx, int gy) {
  const int8_t center = output[gy * coin_model::kOutputGridWidth + gx];
  for (int ny = gy - coin_model::kPeakWindow; ny <= gy + coin_model::kPeakWindow; ++ny) {
    if (ny < 0 || ny >= coin_model::kOutputGridHeight) {
      continue;
    }
    for (int nx = gx - coin_model::kPeakWindow; nx <= gx + coin_model::kPeakWindow; ++nx) {
      if (nx < 0 || nx >= coin_model::kOutputGridWidth) {
        continue;
      }
      if (nx == gx && ny == gy) {
        continue;
      }
      if (output[ny * coin_model::kOutputGridWidth + nx] > center) {
        return false;
      }
    }
  }
  return true;
}

bool is_far_enough(const CoinDetectionResult& result, int gx, int gy) {
  for (int i = 0; i < result.peak_count; ++i) {
    const CoinPeak& peak = result.peaks[i];
    const int dx = abs(peak.gx - gx);
    const int dy = abs(peak.gy - gy);
    if ((dx > dy ? dx : dy) <= coin_model::kPeakMinDistanceCells) {
      return false;
    }
  }
  return true;
}

void sort_peak_array(CoinPeak* peaks, int count) {
  for (int i = 0; i < count - 1; ++i) {
    for (int j = i + 1; j < count; ++j) {
      if (peaks[j].score > peaks[i].score) {
        const CoinPeak temp = peaks[i];
        peaks[i] = peaks[j];
        peaks[j] = temp;
      }
    }
  }
}

void decode_output() {
  constexpr int kMaxCandidates = 48;
  CoinPeak candidates[kMaxCandidates];
  int candidate_count = 0;

  g_last_result.peak_count = 0;
  g_last_result.max_score = 0.0f;
  g_last_result.error = "";
  g_last_result.ok = true;

  const int8_t* output = g_output->data.int8;
  for (int gy = 0; gy < coin_model::kOutputGridHeight; ++gy) {
    for (int gx = 0; gx < coin_model::kOutputGridWidth; ++gx) {
      const int8_t value = output[gy * coin_model::kOutputGridWidth + gx];
      const float score = dequantize_score(value);
      if (score > g_last_result.max_score) {
        g_last_result.max_score = score;
      }
      if (score < coin_model::kDetectionThreshold) {
        continue;
      }
      if (!is_local_peak(output, gx, gy)) {
        continue;
      }

      if (candidate_count >= kMaxCandidates) {
        continue;
      }

      CoinPeak& peak = candidates[candidate_count++];
      peak.gx = gx;
      peak.gy = gy;
      peak.score = score;
    }
  }

  sort_peak_array(candidates, candidate_count);

  const int max_peaks = static_cast<int>(sizeof(g_last_result.peaks) / sizeof(g_last_result.peaks[0]));
  for (int i = 0; i < candidate_count; ++i) {
    const CoinPeak& candidate = candidates[i];
    if (!is_far_enough(g_last_result, candidate.gx, candidate.gy)) {
      continue;
    }
    if (g_last_result.peak_count >= max_peaks) {
      break;
    }
    g_last_result.peaks[g_last_result.peak_count++] = candidate;
  }
}

}  // namespace

namespace coin_detector {

bool begin() {
  if (g_interpreter != nullptr && g_input != nullptr && g_output != nullptr) {
    return true;
  }

  g_model = tflite::GetModel(coin_model::g_coin_model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    set_error("Model schema version mismatch");
    return false;
  }

  if (g_tensor_arena == nullptr) {
    g_tensor_arena = static_cast<uint8_t*>(ps_malloc(kTensorArenaSize));
  }
  if (g_tensor_arena == nullptr) {
    set_error("Failed to allocate tensor arena in PSRAM");
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
      g_model, g_resolver, g_tensor_arena, kTensorArenaSize, g_error_reporter);
  g_interpreter = &static_interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    set_error("AllocateTensors failed");
    return false;
  }

  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);
  if (g_input == nullptr || g_output == nullptr) {
    set_error("Failed to access input or output tensor");
    return false;
  }

  if (g_input->type != kTfLiteInt8 || g_output->type != kTfLiteInt8) {
    set_error("Model tensors are not int8");
    return false;
  }

  if (g_input->bytes != coin_model::kInputTensorBytes) {
    set_error("Unexpected input tensor size");
    return false;
  }

  g_last_error = "";
  g_last_result.arena_used_bytes = g_interpreter->arena_used_bytes();
  g_last_result.arena_size_bytes = kTensorArenaSize;
  g_last_result.ok = true;
  g_last_result.error = "";
  Serial.printf("Detector tensors allocated: %u / %u bytes\n",
                static_cast<unsigned int>(g_last_result.arena_used_bytes),
                static_cast<unsigned int>(g_last_result.arena_size_bytes));
  return true;
}

bool is_ready() {
  return g_interpreter != nullptr && g_input != nullptr && g_output != nullptr;
}

const char* last_error() {
  return g_last_error;
}

int8_t* input_buffer() {
  if (!is_ready()) {
    return nullptr;
  }
  return g_input->data.int8;
}

std::size_t input_bytes() {
  return coin_model::kInputTensorBytes;
}

bool run() {
  if (!is_ready()) {
    set_error("Detector is not ready");
    return false;
  }

  const unsigned long started_at = millis();
  if (g_interpreter->Invoke() != kTfLiteOk) {
    set_error("Invoke failed");
    return false;
  }

  g_last_result.inference_ms = millis() - started_at;
  g_last_result.arena_used_bytes = g_interpreter->arena_used_bytes();
  g_last_result.arena_size_bytes = kTensorArenaSize;
  decode_output();
  g_last_error = "";
  return true;
}

const CoinDetectionResult& last_result() {
  return g_last_result;
}

}  // namespace coin_detector
