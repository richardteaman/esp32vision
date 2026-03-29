#pragma once

#include <cstddef>
#include <cstdint>

struct CoinPeak {
  int gx;
  int gy;
  float score;
};

struct CoinDetectionResult {
  bool ok;
  const char* error;
  int peak_count;
  float max_score;
  unsigned long invoke_ms;
  unsigned long decode_ms;
  unsigned long inference_ms;
  std::size_t arena_used_bytes;
  std::size_t arena_size_bytes;
  CoinPeak peaks[24];
};

namespace coin_detector {

bool begin();
bool is_ready();
const char* last_error();
int8_t* input_buffer();
std::size_t input_bytes();
bool run();
const CoinDetectionResult& last_result();

}  // namespace coin_detector
