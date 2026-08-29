#pragma once

// Copy this file to config.h and fill in real credentials.
// NEVER commit config.h with real credentials.
// If this repo (or any copy of it) is or was under version control / shared anywhere,
// treat the API key, device password, and Wi-Fi password below as exposed and
// rotate them (Firebase Auth password + regenerate/scope the API key, and
// change the Wi-Fi password) — this is unrelated to the PZEM/MUX work below.

#define FIREBASE_API_KEY "REPLACE_WITH_YOUR_API_KEY"
#define FIREBASE_DATABASE_URL "https://YOUR-PROJECT-ID-default-rtdb.asia-southeast1.firebasedatabase.app"
#define FIREBASE_DEVICE_EMAIL "REPLACE_WITH_YOUR_DEVICE_EMAIL"
#define FIREBASE_DEVICE_PASSWORD "REPLACE_WITH_YOUR_DEVICE_PASSWORD"

// Leave blank to provision Wi-Fi through the captive portal. WiFiManager keeps
// successfully provisioned credentials in ESP32 NVS for later boots.
#define WIFI_SSID "REPLACE_WITH_YOUR_WIFI_SSID"
#define WIFI_PASSWORD "REPLACE_WITH_YOUR_WIFI_PASSWORD"

#define ESP32_TX_PIN 17
#define ESP32_RX_PIN 16

// CD74HC4067 analog mux control lines. SIG goes to ESP32_RX_PIN (GPIO16);
// EN is tied to GND in hardware, so it is not driven from firmware.
#define MUX_S0 25
#define MUX_S1 26
#define MUX_S2 27
#define MUX_S3 14

// Delay after driving S0..S3 before the mux output is trusted, and before the
// UART request for the newly-selected PZEM is sent. The CD74HC4067 datasheet
// specifies channel-select propagation on the order of hundreds of ns; 50us
// is a large, safe margin. Raise this only if scope/logic-analyzer evidence
// shows the first request after a channel change is still landing before the
// analog switch has fully settled.
#define MUX_SETTLE_US 50

// PRODUCTION DEFAULT: false. All 9 addresses (see PZEM_COUNT/ADDRESS[] in the
// .ino) are polled, uploaded, and historized independently.
//
// Set to true only as a compile-time diagnostic override to isolate a single
// already-addressed PZEM (e.g. while validating one node on the bus) — set
// PZEM_TEST_ADDRESS to that node's address (1..9) and upload. Do not ship
// with this true; it silently drops the other 8 meters from every Firebase
// write (see selected() in the .ino).
#define PZEM_TEST_MODE false
#define PZEM_TEST_ADDRESS 1

#define PZEM_COUNT 9
#define SERIAL_MONITOR_BAUD 115200
#define PZEM_BAUD 9600

// Timers use millis() only for scheduling, never for RTDB history timestamps.
#define PZEM_POLL_INTERVAL_MS 2500UL
#define LIVE_UPLOAD_INTERVAL_MS 10000UL
#define WIFI_RETRY_INTERVAL_MS 30000UL
#define AUTH_INITIAL_RETRY_MS 15000UL
#define AUTH_MAX_RETRY_MS (15UL * 60UL * 1000UL)
#define CLEANUP_INTERVAL_MS (6UL * 60UL * 60UL * 1000UL)
#define HISTORY_SLOT_SECONDS 300UL
#define HISTORY_RETENTION_SECONDS (60UL * 24UL * 60UL * 60UL)
#define MAX_CLEANUP_BATCHES_PER_CYCLE 5  // max batched DELETE rounds per PZEM per cleanup cycle
#define HTTP_TIMEOUT_MS 8000

// ---------------- Emergency voltage alarm ----------------
// Fires only from a CURRENT VALID reading (readings[i].valid == true). PZEM
// timeout/CRC/MUX failures, Firebase failures, and Wi-Fi failures can never
// reach this - see voltageAbnormal()/checkVoltageAlarm() in the .ino.
#define HIGH_VOLTAGE_LIMIT 250.0f
#define LOW_VOLTAGE_LIMIT 219.0f

// ---------------- Stage 4 fault diagnosis thresholds (match ai.backend values) ----------------
#define FAULT_OVERCURRENT_A 30.0f
#define FAULT_PF_DROP 0.85f
#define FAULT_FREQ_DEVIATION_HZ 2.0f
#define FAULT_HIGH_POWER_W 5000.0f

// Below this, a PZEM's own voltage reading is treated as "this PZEM's AC
// supply is off" rather than a low-voltage fault - e.g. 0V when a load
// switch is off. Must stay well under LOW_VOLTAGE_LIMIT. Tune to your
// installation if loads can legitimately sag below 90V while still "on".
#define AC_PRESENT_VOLTAGE_THRESHOLD 90.0f

// GPIO32/GPIO33 are unused elsewhere in this firmware (confirmed against
// ESP32_TX_PIN/ESP32_RX_PIN/MUX_S0..S3 above) and drive external MOSFET/
// transistor stages only - never the LED strip or buzzer directly.
#define RED_LED_PIN 32
#define BUZZER_PIN 33

// Fixed-duration emergency pattern (Sections 7-8 of the spec). Once an
// alarm event starts it always runs for exactly this long, independent of
// whether the triggering condition clears sooner.
#define ALARM_DURATION_MS 15000UL
#define BUZZER_ON_MS 250UL
#define BUZZER_OFF_MS 250UL
#define LED_ON_MS 250UL
#define LED_OFF_MS 250UL