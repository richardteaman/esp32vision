#include <Arduino.h>
#include <math.h>
#include "esp_camera.h"
#include "img_converters.h"
#include <WiFi.h>
#include <WebServer.h>
#include "coin_model_config.h"
#include "coin_detector.h"

const char* ssid = "home2Ghz_EXT";
const char* password = "5d7030a0e0673640f72f938b67";

WebServer server(80);

enum class CameraMode {
  kStreamJpeg,
  kInferenceRgb565,
};

// ==== AI Thinker ESP32-CAM pin map ====
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
// =====================================

constexpr framesize_t kStreamFrameSize = FRAMESIZE_QVGA;
constexpr int kStreamJpegQuality = 12;
constexpr int kJpegFrameWidth = 320;
constexpr int kJpegFrameHeight = 240;
constexpr std::size_t kJpegRgb888Bytes =
  static_cast<std::size_t>(kJpegFrameWidth) * kJpegFrameHeight * coin_model::kInputChannels;

CameraMode g_camera_mode = CameraMode::kStreamJpeg;
uint8_t* g_inference_rgb888_frame = nullptr;
uint8_t* g_jpeg_rgb888_frame = nullptr;

struct TensorStats {
  int min_value;
  int max_value;
  float mean_value;
};

struct PreprocessTiming {
  unsigned long mode_switch_ms;
  unsigned long capture_ms;
  unsigned long convert_ms;
  unsigned long resize_ms;
  unsigned long quantize_ms;
  unsigned long total_ms;
  int capture_attempts;
};

struct VariantDebugResult {
  const char* name;
  bool swap_rb;
  TensorStats tensor_stats;
  unsigned long quantize_ms;
  CoinDetectionResult detection;
};

static const char* camera_mode_name(CameraMode mode) {
  return mode == CameraMode::kStreamJpeg ? "stream_jpeg" : "inference_rgb565";
}

static bool set_camera_mode(CameraMode mode) {
  if (g_camera_mode == mode) {
    return true;
  }

  sensor_t* sensor = esp_camera_sensor_get();
  if (!sensor) {
    return false;
  }

  int err = 0;
  if (mode == CameraMode::kStreamJpeg) {
    err = sensor->set_pixformat(sensor, PIXFORMAT_JPEG);
    if (err == 0) {
      err = sensor->set_framesize(sensor, kStreamFrameSize);
    }
    if (err == 0) {
      err = sensor->set_quality(sensor, kStreamJpegQuality);
    }
  } else {
    err = sensor->set_pixformat(sensor, PIXFORMAT_RGB565);
    if (err == 0) {
      err = sensor->set_framesize(sensor, FRAMESIZE_96X96);
    }
  }

  if (err != 0) {
    Serial.printf("Failed to switch camera mode to %s\n", camera_mode_name(mode));
    return false;
  }

  delay(150);
  g_camera_mode = mode;
  Serial.printf("Camera mode: %s\n", camera_mode_name(mode));
  return true;
}

static int8_t quantize_input_byte(uint8_t value) {
  const float normalized = static_cast<float>(value) / 255.0f;
  int quantized = static_cast<int>(lroundf(normalized / coin_model::kInputScale)) + coin_model::kInputZeroPoint;
  if (quantized < -128) {
    quantized = -128;
  }
  if (quantized > 127) {
    quantized = 127;
  }
  return static_cast<int8_t>(quantized);
}

static bool ensure_inference_rgb888_frame_buffer() {
  if (g_inference_rgb888_frame != nullptr) {
    return true;
  }

  g_inference_rgb888_frame = static_cast<uint8_t*>(ps_malloc(coin_model::kInputTensorBytes));
  if (g_inference_rgb888_frame == nullptr) {
    g_inference_rgb888_frame = static_cast<uint8_t*>(malloc(coin_model::kInputTensorBytes));
  }
  return g_inference_rgb888_frame != nullptr;
}

static bool ensure_jpeg_rgb888_frame_buffer() {
  if (g_jpeg_rgb888_frame != nullptr) {
    return true;
  }

  g_jpeg_rgb888_frame = static_cast<uint8_t*>(ps_malloc(kJpegRgb888Bytes));
  if (g_jpeg_rgb888_frame == nullptr) {
    g_jpeg_rgb888_frame = static_cast<uint8_t*>(malloc(kJpegRgb888Bytes));
  }
  return g_jpeg_rgb888_frame != nullptr;
}

static void resize_rgb888_nearest(
  const uint8_t* src_buffer,
  int src_width,
  int src_height,
  uint8_t* dst_buffer,
  int dst_width,
  int dst_height
) {
  for (int y = 0; y < dst_height; ++y) {
    const int src_y = y * src_height / dst_height;
    for (int x = 0; x < dst_width; ++x) {
      const int src_x = x * src_width / dst_width;
      const std::size_t src_index =
        (static_cast<std::size_t>(src_y) * src_width + src_x) * coin_model::kInputChannels;
      const std::size_t dst_index =
        (static_cast<std::size_t>(y) * dst_width + x) * coin_model::kInputChannels;
      dst_buffer[dst_index + 0] = src_buffer[src_index + 0];
      dst_buffer[dst_index + 1] = src_buffer[src_index + 1];
      dst_buffer[dst_index + 2] = src_buffer[src_index + 2];
    }
  }
}

static bool capture_rgb888_frame_from_direct_rgb565(uint8_t* rgb_buffer, PreprocessTiming* timing) {
  if (rgb_buffer == nullptr) {
    return false;
  }

  PreprocessTiming local_timing = {};
  const unsigned long mode_switch_started_at = millis();
  if (!set_camera_mode(CameraMode::kInferenceRgb565)) {
    return false;
  }
  local_timing.mode_switch_ms = millis() - mode_switch_started_at;

  bool ok = false;
  for (int attempt = 0; attempt < 3 && !ok; ++attempt) {
    local_timing.capture_attempts += 1;
    const unsigned long capture_started_at = millis();
    camera_fb_t* fb = esp_camera_fb_get();
    local_timing.capture_ms += millis() - capture_started_at;
    if (!fb) {
      delay(30);
      continue;
    }

    const bool frame_ok = fb->format == PIXFORMAT_RGB565 &&
                          fb->width == coin_model::kInputWidth &&
                          fb->height == coin_model::kInputHeight;
    if (frame_ok) {
      const unsigned long convert_started_at = millis();
      ok = fmt2rgb888(
        fb->buf,
        fb->len,
        fb->format,
        rgb_buffer
      );
      local_timing.convert_ms += millis() - convert_started_at;
    }

    esp_camera_fb_return(fb);

    if (!ok) {
      delay(30);
    }
  }

  if (!ok) {
    return false;
  }

  if (timing != nullptr) {
    *timing = local_timing;
  }
  return true;
}

static bool capture_rgb888_frame_from_jpeg_qvga(uint8_t* rgb_buffer, PreprocessTiming* timing) {
  if (rgb_buffer == nullptr ||
      !ensure_inference_rgb888_frame_buffer() ||
      !ensure_jpeg_rgb888_frame_buffer()) {
    return false;
  }

  PreprocessTiming local_timing = {};
  const unsigned long mode_switch_started_at = millis();
  if (!set_camera_mode(CameraMode::kStreamJpeg)) {
    return false;
  }
  local_timing.mode_switch_ms = millis() - mode_switch_started_at;

  bool ok = false;
  for (int attempt = 0; attempt < 3 && !ok; ++attempt) {
    local_timing.capture_attempts += 1;
    const unsigned long capture_started_at = millis();
    camera_fb_t* fb = esp_camera_fb_get();
    local_timing.capture_ms += millis() - capture_started_at;
    if (!fb) {
      delay(30);
      continue;
    }

    const bool frame_ok = fb->format == PIXFORMAT_JPEG &&
                          fb->width == kJpegFrameWidth &&
                          fb->height == kJpegFrameHeight;
    if (frame_ok) {
      const unsigned long convert_started_at = millis();
      ok = fmt2rgb888(
        fb->buf,
        fb->len,
        fb->format,
        g_jpeg_rgb888_frame
      );
      local_timing.convert_ms += millis() - convert_started_at;

      if (ok) {
        const unsigned long resize_started_at = millis();
        resize_rgb888_nearest(
          g_jpeg_rgb888_frame,
          kJpegFrameWidth,
          kJpegFrameHeight,
          rgb_buffer,
          coin_model::kInputWidth,
          coin_model::kInputHeight
        );
        local_timing.resize_ms += millis() - resize_started_at;
      }
    }

    esp_camera_fb_return(fb);

    if (!ok) {
      delay(30);
    }
  }

  if (!ok) {
    return false;
  }

  if (timing != nullptr) {
    *timing = local_timing;
  }
  return true;
}

static void quantize_rgb888_tensor(
  const uint8_t* rgb_buffer,
  int8_t* input_buffer,
  bool swap_rb,
  TensorStats* tensor_stats
) {
  int local_min = 127;
  int local_max = -128;
  long tensor_sum = 0;

  for (std::size_t i = 0; i < coin_model::kInputTensorBytes; ++i) {
    std::size_t source_index = i;
    if (swap_rb && coin_model::kInputChannels == 3) {
      const std::size_t channel = i % coin_model::kInputChannels;
      if (channel == 0) {
        source_index = i + 2;
      } else if (channel == 2) {
        source_index = i - 2;
      }
    }

    const uint8_t raw_value = rgb_buffer[source_index];
    const int8_t quantized = quantize_input_byte(raw_value);
    input_buffer[i] = quantized;
    local_min = min(local_min, static_cast<int>(quantized));
    local_max = max(local_max, static_cast<int>(quantized));
    tensor_sum += quantized;
  }

  if (tensor_stats != nullptr) {
    tensor_stats->min_value = local_min;
    tensor_stats->max_value = local_max;
    tensor_stats->mean_value =
      static_cast<float>(tensor_sum) / static_cast<float>(coin_model::kInputTensorBytes);
  }
}

static bool capture_preprocessed_tensor(
  int8_t* input_buffer,
  TensorStats* tensor_stats,
  PreprocessTiming* timing
) {
  if (input_buffer == nullptr || !ensure_inference_rgb888_frame_buffer()) {
    return false;
  }

  const unsigned long started_at = millis();
  PreprocessTiming local_timing = {};
  if (!capture_rgb888_frame_from_jpeg_qvga(g_inference_rgb888_frame, &local_timing)) {
    return false;
  }

  const unsigned long quantize_started_at = millis();
  quantize_rgb888_tensor(g_inference_rgb888_frame, input_buffer, false, tensor_stats);
  local_timing.quantize_ms = millis() - quantize_started_at;
  local_timing.total_ms = millis() - started_at;

  if (timing != nullptr) {
    *timing = local_timing;
  }
  return true;
}

static bool capture_preprocessed_tensor_direct_rgb565(
  int8_t* input_buffer,
  TensorStats* tensor_stats,
  PreprocessTiming* timing
) {
  if (input_buffer == nullptr || !ensure_inference_rgb888_frame_buffer()) {
    return false;
  }

  const unsigned long started_at = millis();
  PreprocessTiming local_timing = {};
  if (!capture_rgb888_frame_from_direct_rgb565(g_inference_rgb888_frame, &local_timing)) {
    return false;
  }

  const unsigned long quantize_started_at = millis();
  quantize_rgb888_tensor(g_inference_rgb888_frame, input_buffer, false, tensor_stats);
  local_timing.quantize_ms = millis() - quantize_started_at;
  local_timing.total_ms = millis() - started_at;

  if (timing != nullptr) {
    *timing = local_timing;
  }
  return true;
}

static bool run_variant_on_frame(
  const uint8_t* rgb_buffer,
  const char* name,
  bool swap_rb,
  VariantDebugResult* out
) {
  if (rgb_buffer == nullptr || out == nullptr) {
    return false;
  }

  int8_t* input_buffer = coin_detector::input_buffer();
  if (input_buffer == nullptr) {
    return false;
  }

  out->name = name;
  out->swap_rb = swap_rb;

  const unsigned long quantize_started_at = millis();
  quantize_rgb888_tensor(rgb_buffer, input_buffer, swap_rb, &out->tensor_stats);
  out->quantize_ms = millis() - quantize_started_at;

  if (!coin_detector::run()) {
    out->detection = coin_detector::last_result();
    return false;
  }

  out->detection = coin_detector::last_result();
  return true;
}

static void append_tensor_stats_json(String& json, const TensorStats& tensor_stats) {
  json += "{";
  json += "\"min\":";
  json += String(tensor_stats.min_value);
  json += ",";
  json += "\"max\":";
  json += String(tensor_stats.max_value);
  json += ",";
  json += "\"mean\":";
  json += String(tensor_stats.mean_value, 4);
  json += "}";
}

static void append_preprocess_timing_json(String& json, const PreprocessTiming& timing) {
  json += "{";
  json += "\"mode_switch_ms\":";
  json += String(timing.mode_switch_ms);
  json += ",";
  json += "\"capture_ms\":";
  json += String(timing.capture_ms);
  json += ",";
  json += "\"convert_ms\":";
  json += String(timing.convert_ms);
  json += ",";
  json += "\"resize_ms\":";
  json += String(timing.resize_ms);
  json += ",";
  json += "\"quantize_ms\":";
  json += String(timing.quantize_ms);
  json += ",";
  json += "\"total_ms\":";
  json += String(timing.total_ms);
  json += ",";
  json += "\"capture_attempts\":";
  json += String(timing.capture_attempts);
  json += "}";
}

static void append_peaks_json(String& json, const CoinDetectionResult& result) {
  json += "[";
  for (int i = 0; i < result.peak_count; ++i) {
    if (i != 0) {
      json += ",";
    }
    const CoinPeak& peak = result.peaks[i];
    const float center_x =
      (static_cast<float>(peak.gx) + 0.5f) * coin_model::kInputWidth / coin_model::kOutputGridWidth;
    const float center_y =
      (static_cast<float>(peak.gy) + 0.5f) * coin_model::kInputHeight / coin_model::kOutputGridHeight;
    json += "{";
    json += "\"gx\":";
    json += String(peak.gx);
    json += ",";
    json += "\"gy\":";
    json += String(peak.gy);
    json += ",";
    json += "\"score\":";
    json += String(peak.score, 4);
    json += ",";
    json += "\"center_x\":";
    json += String(center_x, 2);
    json += ",";
    json += "\"center_y\":";
    json += String(center_y, 2);
    json += "}";
  }
  json += "]";
}

static void append_variant_debug_json(String& json, const VariantDebugResult& variant) {
  json += "{";
  json += "\"name\":\"";
  json += variant.name;
  json += "\",";
  json += "\"swap_rb\":";
  json += variant.swap_rb ? "true" : "false";
  json += ",";
  json += "\"tensor_stats\":";
  append_tensor_stats_json(json, variant.tensor_stats);
  json += ",";
  json += "\"quantize_ms\":";
  json += String(variant.quantize_ms);
  json += ",";
  json += "\"count\":";
  json += String(variant.detection.peak_count);
  json += ",";
  json += "\"max_score\":";
  json += String(variant.detection.max_score, 4);
  json += ",";
  json += "\"invoke_ms\":";
  json += String(variant.detection.invoke_ms);
  json += ",";
  json += "\"decode_ms\":";
  json += String(variant.detection.decode_ms);
  json += ",";
  json += "\"inference_ms\":";
  json += String(variant.detection.inference_ms);
  json += ",";
  json += "\"peaks\":";
  append_peaks_json(json, variant.detection);
  json += "}";
}

static void handle_root() {
  server.send(200, "text/plain",
              "ESP32-CAM OK\n"
              "GET /capture  -> single JPEG\n"
              "GET /stream   -> MJPEG stream\n"
              "GET /detect/preprocess -> JPEG/QVGA preprocess stats\n"
              "GET /detect/run -> JPEG/QVGA -> resize -> infer\n"
              "GET /detect/run_rgb565 -> legacy direct RGB565 infer\n"
              "GET /detect/debug/channels -> compare RGB vs BGR on one direct RGB565 frame\n");
}

static void handle_capture() {
  if (!set_camera_mode(CameraMode::kStreamJpeg)) {
    server.send(500, "text/plain", "Failed to switch camera to JPEG mode");
    return;
  }
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

static void handle_stream() {
  if (!set_camera_mode(CameraMode::kStreamJpeg)) {
    server.send(500, "text/plain", "Failed to switch camera to JPEG mode");
    return;
  }
  WiFiClient client = server.client();
  client.print(
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
  );

  while (client.connected()) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) break;

    client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    client.write(fb->buf, fb->len);
    client.print("\r\n");
    esp_camera_fb_return(fb);

    delay(50);
  }
}

static void handle_detect_preprocess() {
  int8_t* input_buffer = coin_detector::input_buffer();
  if (input_buffer == nullptr) {
    server.send(500, "text/plain", coin_detector::last_error());
    return;
  }

  TensorStats tensor_stats = {};
  PreprocessTiming preprocess_timing = {};
  if (!capture_preprocessed_tensor(input_buffer, &tensor_stats, &preprocess_timing)) {
    server.send(500, "text/plain", "Failed to convert frame to model input tensor");
    return;
  }

  String sample = "[";
  const std::size_t sample_count =
    coin_model::kInputTensorBytes < 12 ? coin_model::kInputTensorBytes : 12;
  for (std::size_t i = 0; i < sample_count; ++i) {
    if (i != 0) {
      sample += ", ";
    }
    sample += String(static_cast<int>(input_buffer[i]));
  }
  sample += "]";

  String json = "{";
  json += "\"status\":\"prepared\",";
  json += "\"pipeline\":\"jpeg_qvga_resize\",";
  json += "\"camera_mode\":\"";
  json += camera_mode_name(g_camera_mode);
  json += "\",";
  json += "\"input_width\":";
  json += String(coin_model::kInputWidth);
  json += ",";
  json += "\"input_height\":";
  json += String(coin_model::kInputHeight);
  json += ",";
  json += "\"input_channels\":";
  json += String(coin_model::kInputChannels);
  json += ",";
  json += "\"tensor_bytes\":";
  json += String(static_cast<unsigned int>(coin_model::kInputTensorBytes));
  json += ",";
  json += "\"tensor_stats\":";
  append_tensor_stats_json(json, tensor_stats);
  json += ",";
  json += "\"preprocess_timing_ms\":";
  append_preprocess_timing_json(json, preprocess_timing);
  json += ",";
  json += "\"free_heap\":";
  json += String(ESP.getFreeHeap());
  json += ",";
  json += "\"sample\":";
  json += sample;
  json += "}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

static void handle_detect_run() {
  int8_t* input_buffer = coin_detector::input_buffer();
  if (input_buffer == nullptr) {
    server.send(500, "text/plain", coin_detector::last_error());
    return;
  }

  const unsigned long request_started_at = millis();
  TensorStats tensor_stats = {};
  PreprocessTiming preprocess_timing = {};
  if (!capture_preprocessed_tensor(input_buffer, &tensor_stats, &preprocess_timing)) {
    server.send(500, "text/plain", "Failed to convert frame to model input tensor");
    return;
  }

  if (!coin_detector::run()) {
    server.send(500, "text/plain", coin_detector::last_error());
    return;
  }

  const CoinDetectionResult& result = coin_detector::last_result();
  String json = "{";
  json += "\"status\":\"ok\",";
  json += "\"pipeline\":\"jpeg_qvga_resize\",";
  json += "\"camera_mode\":\"";
  json += camera_mode_name(g_camera_mode);
  json += "\",";
  json += "\"count\":";
  json += String(result.peak_count);
  json += ",";
  json += "\"max_score\":";
  json += String(result.max_score, 4);
  json += ",";
  json += "\"tensor_stats\":";
  append_tensor_stats_json(json, tensor_stats);
  json += ",";
  json += "\"preprocess_timing_ms\":";
  append_preprocess_timing_json(json, preprocess_timing);
  json += ",";
  json += "\"invoke_ms\":";
  json += String(result.invoke_ms);
  json += ",";
  json += "\"decode_ms\":";
  json += String(result.decode_ms);
  json += ",";
  json += "\"inference_ms\":";
  json += String(result.inference_ms);
  json += ",";
  json += "\"request_total_ms\":";
  json += String(millis() - request_started_at);
  json += ",";
  json += "\"arena_used_bytes\":";
  json += String(static_cast<unsigned int>(result.arena_used_bytes));
  json += ",";
  json += "\"arena_size_bytes\":";
  json += String(static_cast<unsigned int>(result.arena_size_bytes));
  json += ",";
  json += "\"peaks\":";
  append_peaks_json(json, result);
  json += "}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

static void handle_detect_run_rgb565() {
  int8_t* input_buffer = coin_detector::input_buffer();
  if (input_buffer == nullptr) {
    server.send(500, "text/plain", coin_detector::last_error());
    return;
  }

  const unsigned long request_started_at = millis();
  TensorStats tensor_stats = {};
  PreprocessTiming preprocess_timing = {};
  if (!capture_preprocessed_tensor_direct_rgb565(input_buffer, &tensor_stats, &preprocess_timing)) {
    server.send(500, "text/plain", "Failed to convert direct RGB565 frame to model input tensor");
    return;
  }

  if (!coin_detector::run()) {
    server.send(500, "text/plain", coin_detector::last_error());
    return;
  }

  const CoinDetectionResult& result = coin_detector::last_result();
  String json = "{";
  json += "\"status\":\"ok\",";
  json += "\"pipeline\":\"direct_rgb565\",";
  json += "\"camera_mode\":\"";
  json += camera_mode_name(g_camera_mode);
  json += "\",";
  json += "\"count\":";
  json += String(result.peak_count);
  json += ",";
  json += "\"max_score\":";
  json += String(result.max_score, 4);
  json += ",";
  json += "\"tensor_stats\":";
  append_tensor_stats_json(json, tensor_stats);
  json += ",";
  json += "\"preprocess_timing_ms\":";
  append_preprocess_timing_json(json, preprocess_timing);
  json += ",";
  json += "\"invoke_ms\":";
  json += String(result.invoke_ms);
  json += ",";
  json += "\"decode_ms\":";
  json += String(result.decode_ms);
  json += ",";
  json += "\"inference_ms\":";
  json += String(result.inference_ms);
  json += ",";
  json += "\"request_total_ms\":";
  json += String(millis() - request_started_at);
  json += ",";
  json += "\"arena_used_bytes\":";
  json += String(static_cast<unsigned int>(result.arena_used_bytes));
  json += ",";
  json += "\"arena_size_bytes\":";
  json += String(static_cast<unsigned int>(result.arena_size_bytes));
  json += ",";
  json += "\"peaks\":";
  append_peaks_json(json, result);
  json += "}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

static void handle_detect_debug_channels() {
  if (!coin_detector::is_ready()) {
    server.send(500, "text/plain", coin_detector::last_error());
    return;
  }
  if (!ensure_inference_rgb888_frame_buffer()) {
    server.send(500, "text/plain", "Failed to allocate RGB888 debug buffer");
    return;
  }

  const unsigned long request_started_at = millis();
  PreprocessTiming capture_timing = {};
  if (!capture_rgb888_frame_from_direct_rgb565(g_inference_rgb888_frame, &capture_timing)) {
    server.send(500, "text/plain", "Failed to capture RGB888 debug frame");
    return;
  }
  capture_timing.total_ms =
    capture_timing.mode_switch_ms + capture_timing.capture_ms + capture_timing.convert_ms;

  VariantDebugResult native_rgb = {};
  VariantDebugResult native_bgr = {};
  if (!run_variant_on_frame(g_inference_rgb888_frame, "native_rgb", false, &native_rgb)) {
    server.send(500, "text/plain", coin_detector::last_error());
    return;
  }
  if (!run_variant_on_frame(g_inference_rgb888_frame, "native_bgr", true, &native_bgr)) {
    server.send(500, "text/plain", coin_detector::last_error());
    return;
  }

  String json = "{";
  json += "\"status\":\"ok\",";
  json += "\"camera_mode\":\"";
  json += camera_mode_name(g_camera_mode);
  json += "\",";
  json += "\"capture_timing_ms\":";
  append_preprocess_timing_json(json, capture_timing);
  json += ",";
  json += "\"request_total_ms\":";
  json += String(millis() - request_started_at);
  json += ",";
  json += "\"free_heap\":";
  json += String(ESP.getFreeHeap());
  json += ",";
  json += "\"variants\":[";
  append_variant_debug_json(json, native_rgb);
  json += ",";
  append_variant_debug_json(json, native_bgr);
  json += "]}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\nBooting...");

  Serial.printf("PSRAM found: %s\n", psramFound() ? "YES" : "NO");
  Serial.printf("PSRAM size: %u\n", ESP.getPsramSize());
  Serial.printf("Free heap: %u\n", ESP.getFreeHeap());

  if (coin_detector::begin()) {
    const CoinDetectionResult& detector_state = coin_detector::last_result();
    Serial.printf("Detector ready, arena used: %u / %u bytes\n",
                  static_cast<unsigned int>(detector_state.arena_used_bytes),
                  static_cast<unsigned int>(detector_state.arena_size_bytes));
  } else {
    Serial.printf("Detector init failed: %s\n", coin_detector::last_error());
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size   = kStreamFrameSize; // 320x240 стабильно
  config.jpeg_quality = kStreamJpegQuality;
  config.fb_count     = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    while (true) delay(1000);
  }
  Serial.println("Camera OK");

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.printf("Connecting to WiFi: %s\n", ssid);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  server.on("/", handle_root);
  server.on("/capture", HTTP_GET, handle_capture);
  server.on("/stream", HTTP_GET, handle_stream);
  server.on("/detect/preprocess", HTTP_GET, handle_detect_preprocess);
  server.on("/detect/run", HTTP_GET, handle_detect_run);
  server.on("/detect/run_rgb565", HTTP_GET, handle_detect_run_rgb565);
  server.on("/detect/debug/channels", HTTP_GET, handle_detect_debug_channels);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}
