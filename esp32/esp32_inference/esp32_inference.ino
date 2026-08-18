/*
 * ESP32 TinyML Virtual Temperature Sensor — Pure C Inference
 * Model: MLP [4→32→16→1], Float32 weights
 * No TFLite library needed!
 *
 * Serial protocol:
 *   Input:  "f1,f2,f3,f4\n" (4 raw feature values)
 *   Output: "pred,latency_us,free_heap\n"
 *   "INFO\n" → prints model info
 */

#include "model_weights.h"   // W1,B1,W2,B2,W3,B3, sc_mean, sc_scale

// Buffers
float scaled[N_FEATURES];
float h1[HIDDEN1];
float h2[HIDDEN2];
float output;

// Stats
uint32_t inference_count = 0;
uint32_t total_latency_us = 0;
uint32_t min_latency_us = UINT32_MAX;
uint32_t max_latency_us = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  
  Serial.println("=== ESP32 TinyML Virtual T Sensor (Pure C) ===");
  Serial.printf("Model: MLP[4->32->16->1], %d params, %d bytes\n", 705, 705*4);
  Serial.printf("Free heap: %u bytes\n", ESP.getFreeHeap());
  Serial.println("READY");
}

float run_inference(float* x) {
  // Layer 1: Dense(4→32, ReLU)
  for (int j = 0; j < HIDDEN1; j++) {
    float sum = b1[j];
    for (int i = 0; i < N_FEATURES; i++) {
      sum += x[i] * w1[i * HIDDEN1 + j];  // W1 is (4, 32) row-major
    }
    h1[j] = sum > 0.0f ? sum : 0.0f;  // ReLU
  }
  
  // Layer 2: Dense(32→16, ReLU)
  for (int j = 0; j < HIDDEN2; j++) {
    float sum = b2[j];
    for (int i = 0; i < HIDDEN1; i++) {
      sum += h1[i] * w2[i * HIDDEN2 + j];  // W2 is (32, 16) row-major
    }
    h2[j] = sum > 0.0f ? sum : 0.0f;
  }
  
  // Layer 3: Dense(16→1, Linear)
  float out = b3[0];
  for (int i = 0; i < HIDDEN2; i++) {
    out += h2[i] * w3[i];  // W3 is (16, 1)
  }
  
  return out;
}

void process_line(String line) {
  line.trim();
  
  if (line == "INFO") {
    Serial.printf("MODEL: MLP[4->32->16->1] F32, %d params, %d bytes\n", 705, 705*4);
    Serial.printf("HEAP: %u bytes free\n", ESP.getFreeHeap());
    Serial.printf("INFERENCES: %u, AVG_LAT: %u us\n", 
                  inference_count,
                  inference_count > 0 ? total_latency_us / inference_count : 0);
    if (inference_count > 0) {
      Serial.printf("LAT_RANGE: %u - %u us\n", min_latency_us, max_latency_us);
    }
    return;
  }
  
  // Parse 4 comma-separated floats (raw sensor values)
  float raw[4];
  int idx = 0;
  int start = 0;
  for (int i = 0; i <= (int)line.length() && idx < 4; i++) {
    if (i == (int)line.length() || line.charAt(i) == ',') {
      raw[idx++] = line.substring(start, i).toFloat();
      start = i + 1;
    }
  }
  
  if (idx != 4) {
    Serial.println("ERR:need 4 values");
    return;
  }
  
  // StandardScaler: x_scaled = (x - mean) / scale
  for (int i = 0; i < N_FEATURES; i++) {
    scaled[i] = (raw[i] - sc_mean[i]) / sc_scale[i];
  }
  
  // Inference with timing
  uint32_t t0 = micros();
  float pred = run_inference(scaled);
  uint32_t latency = micros() - t0;
  
  // Stats
  inference_count++;
  total_latency_us += latency;
  if (latency < min_latency_us) min_latency_us = latency;
  if (latency > max_latency_us) max_latency_us = latency;
  
  Serial.printf("%.4f,%u,%u\n", pred, latency, ESP.getFreeHeap());
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    process_line(line);
  }
}
