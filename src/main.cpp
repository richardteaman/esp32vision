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

CameraMode g_camera_mode = CameraMode::kStreamJpeg;

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

static bool capture_preprocessed_tensor(int8_t* input_buffer, int* tensor_min, int* tensor_max, float* tensor_mean) {
  if (!set_camera_mode(CameraMode::kInferenceRgb565)) {
    return false;
  }

  bool ok = false;
  int local_min = 127;
  int local_max = -128;
  long tensor_sum = 0;

  for (int attempt = 0; attempt < 3 && !ok; ++attempt) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      delay(30);
      continue;
    }

    const bool frame_ok = fb->format == PIXFORMAT_RGB565 &&
                          fb->width == coin_model::kInputWidth &&
                          fb->height == coin_model::kInputHeight;
    if (frame_ok) {
      ok = fmt2rgb888(
        fb->buf,
        fb->len,
        fb->format,
        reinterpret_cast<uint8_t*>(input_buffer)
      );
    }

    esp_camera_fb_return(fb);

    if (!ok) {
      delay(30);
    }
  }

  if (!ok) {
    return false;
  }

  for (std::size_t i = 0; i < coin_model::kInputTensorBytes; ++i) {
    const uint8_t raw_value = static_cast<uint8_t>(input_buffer[i]);
    const int8_t quantized = quantize_input_byte(raw_value);
    input_buffer[i] = quantized;
    local_min = min(local_min, static_cast<int>(quantized));
    local_max = max(local_max, static_cast<int>(quantized));
    tensor_sum += quantized;
  }

  if (tensor_min != nullptr) {
    *tensor_min = local_min;
  }
  if (tensor_max != nullptr) {
    *tensor_max = local_max;
  }
  if (tensor_mean != nullptr) {
    *tensor_mean = static_cast<float>(tensor_sum) / static_cast<float>(coin_model::kInputTensorBytes);
  }
  return true;
}

static void handle_root() {
  server.send(200, "text/plain",
              "ESP32-CAM OK\n"
              "GET /capture  -> single JPEG\n"
              "GET /stream   -> MJPEG stream\n"
              "GET /detect/preprocess -> capture and prepare int8 tensor\n");
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

  int tensor_min = 0;
  int tensor_max = 0;
  float tensor_mean = 0.0f;
  if (!capture_preprocessed_tensor(input_buffer, &tensor_min, &tensor_max, &tensor_mean)) {
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
  json += "\"tensor_min\":";
  json += String(tensor_min);
  json += ",";
  json += "\"tensor_max\":";
  json += String(tensor_max);
  json += ",";
  json += "\"tensor_mean\":";
  json += String(tensor_mean, 4);
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

  if (!capture_preprocessed_tensor(input_buffer, nullptr, nullptr, nullptr)) {
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
  json += "\"camera_mode\":\"";
  json += camera_mode_name(g_camera_mode);
  json += "\",";
  json += "\"count\":";
  json += String(result.peak_count);
  json += ",";
  json += "\"max_score\":";
  json += String(result.max_score, 4);
  json += ",";
  json += "\"inference_ms\":";
  json += String(result.inference_ms);
  json += ",";
  json += "\"arena_used_bytes\":";
  json += String(static_cast<unsigned int>(result.arena_used_bytes));
  json += ",";
  json += "\"arena_size_bytes\":";
  json += String(static_cast<unsigned int>(result.arena_size_bytes));
  json += ",";
  json += "\"peaks\":[";
  for (int i = 0; i < result.peak_count; ++i) {
    if (i != 0) {
      json += ",";
    }
    const CoinPeak& peak = result.peaks[i];
    const float center_x = (static_cast<float>(peak.gx) + 0.5f) * coin_model::kInputWidth / coin_model::kOutputGridWidth;
    const float center_y = (static_cast<float>(peak.gy) + 0.5f) * coin_model::kInputHeight / coin_model::kOutputGridHeight;
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
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}
