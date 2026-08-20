#pragma once

// Copy this project outside source control before entering real credentials.
// NOTE: this file currently contains live Firebase/Wi-Fi credentials. If this
// repo (or any copy of it) is or was under version control / shared anywhere,
// treat the API key, device password, and Wi-Fi password below as exposed and
// rotate them (Firebase Auth password + regenerate/scope the API key, and
// change the Wi-Fi password) — this is unrelated to the PZEM/MUX work below.
#define FIREBASE_API_KEY "AIzaSyDCQJgHdIb5CkGRhAPOI-ynVfHdNFSo6bs"
#define FIREBASE_DATABASE_URL "https://smart-energy-monitoring-5a2a4-default-rtdb.asia-southeast1.firebasedatabase.app"
#define FIREBASE_DEVICE_EMAIL "smartenergymonitoringsystem28@gmail.com"
#define FIREBASE_DEVICE_PASSWORD "Group@02"

// Leave blank to provision Wi-Fi through the captive portal. WiFiManager keeps
// successfully provisioned credentials in ESP32 NVS for later boots.
#define WIFI_SSID "vivoT2x"
#define WIFI_PASSWORD "anshul@809"

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
#define PZEM_TEST_MODE true
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
#define HTTP_TIMEOUT_MS 8000
