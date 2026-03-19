#pragma once

// Auto-generated from: /home/richt/Documents/PlatformIO/Projects/esp32vision/ml/outputs/baseline_rgb_hard_ref/model_int8.tflite

#include <cstddef>
#include <cstdint>

namespace coin_model {

constexpr int kInputHeight = 96;
constexpr int kInputWidth = 96;
constexpr int kInputChannels = 3;
constexpr std::size_t kInputTensorBytes = static_cast<std::size_t>(kInputHeight) * kInputWidth * kInputChannels;

constexpr int kOutputGridHeight = 12;
constexpr int kOutputGridWidth = 12;
constexpr int kOutputChannels = 1;
constexpr std::size_t kOutputTensorBytes = static_cast<std::size_t>(kOutputGridHeight) * kOutputGridWidth * kOutputChannels;

constexpr float kInputScale = 0.0039215689f;
constexpr int kInputZeroPoint = -128;
constexpr float kOutputScale = 0.0039062500f;
constexpr int kOutputZeroPoint = -128;

constexpr float kDetectionThreshold = 0.3500f;
constexpr int kPeakWindow = 1;
constexpr int kPeakMinDistanceCells = 2;
constexpr int kMatchRadiusCells = 1;

}  // namespace coin_model
