/*
  Smart Energy Monitor — ESP32 DevKit V1 / PZEM-004T V4
  Dashboard contract: /meters/pzem_N and /history/pzem_N/<unix-seconds>.

  HARDWARE EVIDENCE: this PZEM at address 7 works through a USB-to-TTL adapter
  but produces no valid UART bytes when wired directly to GPIO16/17. That
  isolates the fault to the ESP32/PZEM logic-level interface, not the PZEM,
  its address, this library, or its measurement side. Firmware cannot correct
  a voltage-level mismatch. Use a validated bidirectional level shifter.

  IMPORTANT: do not use the proposed diode-OR PZEM-TX-to-ESP32-RX bus. Its
  direction blocks UART LOW/start bits and it does not solve voltage-level
  compatibility. Do not directly parallel multiple push-pull PZEM TX lines.

  MUX UPDATE: GPIO16 (ESP32 RX2) is now fed by a CD74HC4067 analog mux SIG
  output instead of being wired to all 9 PZEM TX lines directly. Only one
  PZEM's TX reaches the ESP32 at a time; selectPzemMux() picks which one
  before each read. GPIO17 (ESP32 TX2) is unchanged and still fans out to
  all 9 PZEM RX pins, since the PZEM library addresses requests per-unit and
  only the addressed PZEM replies.
*/

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <WiFiManager.h>
#include <PZEM004Tv30.h>
#include <time.h>
#include "config.h"

struct Reading {
  bool valid = false;
  float voltage = NAN, current = NAN, power = NAN, energy = NAN;
  float frequency = NAN, pf = NAN;
  time_t seenAt = 0;
};

enum AuthState : uint8_t { AUTH_IDLE, AUTH_PENDING, AUTH_READY };
// Defined before all functions so Arduino IDE prototype generation can never
// emit a declaration that references an unknown return type.
enum PzemRawResult : uint8_t { PZEM_RAW_NO_BYTES, PZEM_RAW_INVALID_FRAME, PZEM_RAW_UNEXPECTED_FORMAT, PZEM_RAW_VALID };

WiFiManager wifiManager;
PZEM004Tv30 pzems[PZEM_COUNT];
Reading readings[PZEM_COUNT];

AuthState authState = AUTH_IDLE;
String idToken, refreshToken;
time_t tokenExpiresAt = 0;
uint32_t authRetryMs = AUTH_INITIAL_RETRY_MS;
uint32_t lastAuthAttemptMs = 0, lastPollMs = 0, lastLiveUploadMs = 0;
uint32_t lastWiFiRetryMs = 0, lastCleanupMs = 0;
time_t lastHistorySlot = 0;
bool portalRunning = false;

const uint8_t ADDRESS[PZEM_COUNT] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

bool elapsed(uint32_t now, uint32_t then, uint32_t period) {
  return (uint32_t)(now - then) >= period;
}

bool validClock() { return time(nullptr) >= 1700000000; }

bool selected(uint8_t index) {
  return !PZEM_TEST_MODE || ADDRESS[index] == PZEM_TEST_ADDRESS;
}

// Selects which PZEM's TX line reaches ESP32_RX_PIN through the CD74HC4067.
// channel maps 1:1 to PZEM array index i (index 0 -> mux C0 -> PZEM1, ...,
// index 8 -> mux C8 -> PZEM9), matching the physical wiring. EN is tied to
// GND in hardware and is not driven here. A short settle delay is included
// because the mux is switched immediately before a request/response UART
// transaction, not during one.
void selectPzemMux(uint8_t channel) {
  digitalWrite(MUX_S0, channel & 0x01);
  digitalWrite(MUX_S1, (channel >> 1) & 0x01);
  digitalWrite(MUX_S2, (channel >> 2) & 0x01);
  digitalWrite(MUX_S3, (channel >> 3) & 0x01);
  delayMicroseconds(50); // CD74HC4067 channel-select propagation margin
}

bool jsonWhitespace(char value) {
  return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

// Locates the first value character after a JSON object's named field. This
// intentionally tolerates whitespace around the colon without logging JSON.
int jsonValueStart(const String &json, const char *key) {
  const String field = String("\"") + key + "\"";
  const int fieldAt = json.indexOf(field);
  if (fieldAt < 0) return -1;
  int cursor = json.indexOf(':', fieldAt + field.length());
  if (cursor < 0) return -1;
  while (++cursor < (int)json.length() && jsonWhitespace(json[cursor])) {}
  return cursor < (int)json.length() ? cursor : -1;
}

String jsonStringField(const String &json, const char *key) {
  const int start = jsonValueStart(json, key);
  if (start < 0 || json[start] != '"') return "";
  const int end = json.indexOf('"', start + 1);
  return end < 0 ? "" : json.substring(start + 1, end);
}

// Firebase can return expiry fields either as JSON numbers or quoted numeric
// strings. Both forms, with optional whitespace, are accepted.
long jsonNumberField(const String &json, const char *key) {
  int cursor = jsonValueStart(json, key);
  if (cursor < 0) return 0;
  const bool quoted = json[cursor] == '"';
  if (quoted) ++cursor;
  const int start = cursor;
  while (cursor < (int)json.length() && isDigit(json[cursor])) ++cursor;
  if (cursor == start || (quoted && (cursor >= (int)json.length() || json[cursor] != '"'))) return 0;
  return json.substring(start, cursor).toInt();
}

void reportFirebaseHttpError(int httpCode, const String &reply) {
  Serial.printf("[FIREBASE] HTTP status: %d\n", httpCode);
  Serial.printf("[FIREBASE] HTTP error: %d\n", httpCode);
  const long errorCode = jsonNumberField(reply, "code");
  const String errorMessage = jsonStringField(reply, "message");
  if (errorCode) Serial.printf("[FIREBASE] Error code: %ld\n", errorCode);
  if (errorMessage.length()) Serial.printf("[FIREBASE] Error message: %s\n", errorMessage.c_str());
}

void reportFirebaseTokenFields(const String &token, const String &refresh, long expires) {
  Serial.printf("[FIREBASE] idToken present: %s\n", token.length() ? "YES" : "NO");
  Serial.printf("[FIREBASE] refreshToken present: %s\n", refresh.length() ? "YES" : "NO");
  Serial.printf("[FIREBASE] expiresIn parsed: %ld\n", expires);
}

// All HTTP calls have a bounded timeout. They are invoked only from scheduled
// tasks; sensor polling never waits for an authentication retry loop.
int httpsRequest(const String &url, const char *method, const String &body, String *response = nullptr) {
  if (WiFi.status() != WL_CONNECTED) return -1;
  WiFiClientSecure client;
  // TODO before production: configure a current Google CA bundle for this
  // client. setInsecure is used only so the first hardware/Firebase test works
  // on Arduino-ESP32 installations without a certificate bundle.
  client.setInsecure();
  HTTPClient https;
  https.setConnectTimeout(HTTP_TIMEOUT_MS);
  https.setTimeout(HTTP_TIMEOUT_MS);
  if (!https.begin(client, url)) return -2;
  https.addHeader("Content-Type", "application/json");
  int code = -3;
  if (!strcmp(method, "POST")) code = https.POST(body);
  else if (!strcmp(method, "PUT")) code = https.PUT(body);
  else if (!strcmp(method, "PATCH")) code = https.PATCH(body);
  else if (!strcmp(method, "DELETE")) code = https.sendRequest("DELETE");
  else if (!strcmp(method, "GET")) code = https.GET();
  if (response && code > 0) *response = https.getString();
  https.end();
  return code;
}

String authUrl(const char *path) {
  return String("https://identitytoolkit.googleapis.com/v1/") + path + "?key=" + FIREBASE_API_KEY;
}

String refreshUrl() {
  return String("https://securetoken.googleapis.com/v1/token?key=") + FIREBASE_API_KEY;
}

String dbUrl(const String &path) {
  return String(FIREBASE_DATABASE_URL) + path + (idToken.length() ? "?auth=" + idToken : "");
}

void markAuthFailed(const char *reason) {
  Serial.printf("[FIREBASE] Authentication failed: %s; retry in %lu s\n", reason, authRetryMs / 1000UL);
  idToken = "";
  tokenExpiresAt = 0;
  authState = AUTH_IDLE;
  authRetryMs = min(authRetryMs * 2UL, AUTH_MAX_RETRY_MS);
}

bool signIn() {
  if (String(FIREBASE_DEVICE_PASSWORD) == "REPLACE_WITH_DEVICE_PASSWORD") {
    Serial.println("[FIREBASE] Set FIREBASE_DEVICE_PASSWORD in config.h");
    return false;
  }
  String reply;
  String body = String("{\"email\":\"") + FIREBASE_DEVICE_EMAIL + "\",\"password\":\"" + FIREBASE_DEVICE_PASSWORD + "\",\"returnSecureToken\":true}";
  int code = httpsRequest(authUrl("accounts:signInWithPassword"), "POST", body, &reply);
  if (code < 200 || code >= 300) { reportFirebaseHttpError(code, reply); markAuthFailed("sign-in request"); return false; }
  idToken = jsonStringField(reply, "idToken");
  refreshToken = jsonStringField(reply, "refreshToken");
  long expires = jsonNumberField(reply, "expiresIn");
  Serial.printf("[FIREBASE] HTTP status: %d\n", code);
  reportFirebaseTokenFields(idToken, refreshToken, expires);
  if (!idToken.length() || !refreshToken.length() || expires <= 0) { markAuthFailed("invalid sign-in response"); return false; }
  tokenExpiresAt = time(nullptr) + expires;
  authState = AUTH_READY;
  authRetryMs = AUTH_INITIAL_RETRY_MS;
  Serial.println("[FIREBASE] Authentication successful");
  return true;
}

bool refreshAuthToken() {
  String reply;
  String body = String("grant_type=refresh_token&refresh_token=") + refreshToken;
  // OAuth token exchange uses form encoding.
  if (WiFi.status() != WL_CONNECTED) return false;
  WiFiClientSecure client;
  client.setInsecure(); // Replace with a maintained CA bundle before production.
  HTTPClient https;
  https.setConnectTimeout(HTTP_TIMEOUT_MS); https.setTimeout(HTTP_TIMEOUT_MS);
  if (!https.begin(client, refreshUrl())) return false;
  https.addHeader("Content-Type", "application/x-www-form-urlencoded");
  int code = https.POST(body);
  if (code > 0) reply = https.getString();
  https.end();
  if (code < 200 || code >= 300) { reportFirebaseHttpError(code, reply); markAuthFailed("token refresh"); return false; }
  idToken = jsonStringField(reply, "id_token");
  refreshToken = jsonStringField(reply, "refresh_token");
  long expires = jsonNumberField(reply, "expires_in");
  Serial.printf("[FIREBASE] HTTP status: %d\n", code);
  reportFirebaseTokenFields(idToken, refreshToken, expires);
  if (!idToken.length() || !refreshToken.length() || expires <= 0) { markAuthFailed("invalid refresh response"); return false; }
  tokenExpiresAt = time(nullptr) + expires;
  authState = AUTH_READY;
  Serial.println("[FIREBASE] Token refreshed");
  return true;
}

void serviceAuthentication(uint32_t now) {
  if (WiFi.status() != WL_CONNECTED) return;
  if (authState == AUTH_READY && validClock() && time(nullptr) < tokenExpiresAt - 300) return;
  if (!elapsed(now, lastAuthAttemptMs, authRetryMs)) return;
  lastAuthAttemptMs = now;
  Serial.println(authState == AUTH_READY ? "[FIREBASE] Refreshing token" : "[FIREBASE] Authenticating");
  if (authState == AUTH_READY && refreshToken.length()) refreshAuthToken();
  else signIn();
}

void syncClock() {
  if (WiFi.status() != WL_CONNECTED || validClock()) return;
  configTime(0, 0, "time.google.com", "pool.ntp.org", "time.nist.gov");
  Serial.println("[TIME] NTP sync requested");
}

String readingJson(const Reading &r, time_t timestamp, bool history) {
  char data[280];
  snprintf(data, sizeof(data), "{\"voltage\":%.1f,\"current\":%.2f,\"power\":%.1f,\"energy\":%.3f,\"frequency\":%.1f,\"pf\":%.2f,\"status\":\"online\",\"timestamp\":%lld}",
    r.voltage, r.current, r.power, r.energy, r.frequency, r.pf, (long long)timestamp);
  String out(data);
  if (!history) out = out.substring(0, out.length() - 1) + ",\"lastSeen\":" + String((long long)timestamp) + "}";
  return out;
}

uint16_t modbusCrc(const uint8_t *data, size_t length) {
  uint16_t crc = 0xFFFF;
  while (length--) {
    crc ^= *data++;
    for (uint8_t bit = 0; bit < 8; ++bit) crc = (crc & 1) ? (crc >> 1) ^ 0xA001 : crc >> 1;
  }
  return crc;
}

// Sends the same read-only 0x04 register request used by PZEM004Tv30. It is
// invoked only after a library read fails and never changes a PZEM setting.
PzemRawResult probePzemRaw(uint8_t address, uint8_t *reply, size_t &replyLength) {
  uint8_t request[8] = { address, 0x04, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00 };
  const uint16_t crc = modbusCrc(request, 6);
  request[6] = crc & 0xFF;
  request[7] = crc >> 8;
  while (Serial2.available()) Serial2.read();
  Serial2.write(request, sizeof(request));
  Serial2.flush();

  replyLength = 0;
  const uint32_t started = millis();
  while ((uint32_t)(millis() - started) < 200 && replyLength < 32) {
    while (Serial2.available() && replyLength < 32) reply[replyLength++] = Serial2.read();
    delay(1);
  }
  if (replyLength == 0) return PZEM_RAW_NO_BYTES;
  const bool crcOk = replyLength >= 2 && modbusCrc(reply, replyLength - 2) == ((uint16_t)reply[replyLength - 2] | ((uint16_t)reply[replyLength - 1] << 8));
  if (!crcOk || reply[0] != address || reply[1] != 0x04) return PZEM_RAW_INVALID_FRAME;
  return (replyLength == 25 && reply[2] == 20) ? PZEM_RAW_VALID : PZEM_RAW_UNEXPECTED_FORMAT;
}

void diagnosePzemFailure(uint8_t address) {
  uint8_t reply[32];
  size_t length = 0;
  switch (probePzemRaw(address, reply, length)) {
    case PZEM_RAW_NO_BYTES:
      Serial.printf("[PZEM %u] DIAG: 0 UART bytes. With USB-TTL verified, this indicates the direct ESP32/PZEM logic-level interface or wiring.\n", address);
      break;
    case PZEM_RAW_INVALID_FRAME:
      Serial.printf("[PZEM %u] DIAG: %u invalid UART byte(s), indicating noise, logic-level mismatch, or wiring.\n", address, (unsigned)length);
      break;
    case PZEM_RAW_UNEXPECTED_FORMAT:
      Serial.printf("[PZEM %u] DIAG: CRC-valid frame with unexpected format (%u bytes).\n", address, (unsigned)length);
      break;
    case PZEM_RAW_VALID:
      Serial.printf("[PZEM %u] DIAG: valid raw frame; library timeout was transient.\n", address);
      break;
  }
}

void readMeter(uint8_t i) {
  selectPzemMux(i); // route this PZEM's TX to ESP32_RX_PIN before any UART transaction for it
  Serial.printf("[POLL] PZEM %u\n", ADDRESS[i]);
  Reading next;
  next.voltage = pzems[i].voltage();
  next.current = pzems[i].current();
  next.power = pzems[i].power();
  next.energy = pzems[i].energy();
  next.frequency = pzems[i].frequency();
  next.pf = pzems[i].pf();
  next.valid = isfinite(next.voltage) && isfinite(next.current) && isfinite(next.power) && isfinite(next.energy) && isfinite(next.frequency) && isfinite(next.pf);
  if (next.valid) {
    next.seenAt = validClock() ? time(nullptr) : 0;
    readings[i] = next;
    Serial.printf("[PZEM %u] OK V=%.1f I=%.2f P=%.1f E=%.3f F=%.1f PF=%.2f\n", ADDRESS[i], next.voltage, next.current, next.power, next.energy, next.frequency, next.pf);
  } else {
    readings[i].valid = false;
    Serial.printf("[PZEM %u] Library read failed; running one bounded raw diagnostic.\n", ADDRESS[i]);
    diagnosePzemFailure(ADDRESS[i]); // mux channel from selectPzemMux(i) above is still selected here
  }
}

void pollMeters(uint32_t now) {
  if (!elapsed(now, lastPollMs, PZEM_POLL_INTERVAL_MS)) return;
  lastPollMs = now;
  const uint32_t cycleStart = millis();
  uint8_t attempted = 0, valid = 0;
  for (uint8_t i = 0; i < PZEM_COUNT; ++i) {
    if (!selected(i)) continue;
    ++attempted;
    readMeter(i);
    if (readings[i].valid) ++valid;
  }
  const uint32_t cycleMs = millis() - cycleStart;
  Serial.printf("[POLL] Cycle complete: %u/%u valid (%lu ms)\n", valid, attempted, (unsigned long)cycleMs);
  if (cycleMs >= LIVE_UPLOAD_INTERVAL_MS) {
    Serial.println("[POLL] WARNING: cycle time reached the live-upload interval; Wi-Fi/Firebase servicing may be delayed. Consider raising LIVE_UPLOAD_INTERVAL_MS or investigating slow/unresponsive addresses.");
  }
}

String multiPathJson(const char *root, time_t slot) {
  String body = "{";
  bool first = true;
  for (uint8_t i = 0; i < PZEM_COUNT; ++i) {
    if (!selected(i) || !readings[i].valid) continue;
    if (!first) body += ',';
    String path = String(root) + "/pzem_" + ADDRESS[i] + (slot ? "/" + String((long long)slot) : "");
    body += "\"" + path + "\":" + readingJson(readings[i], slot ? slot : time(nullptr), slot != 0);
    first = false;
  }
  return body + "}";
}

void uploadLive(uint32_t now) {
  if (!elapsed(now, lastLiveUploadMs, LIVE_UPLOAD_INTERVAL_MS) || authState != AUTH_READY || !validClock()) return;
  lastLiveUploadMs = now;
  String body = multiPathJson("meters", 0);
  if (body == "{}") return;
  int code = httpsRequest(dbUrl("/.json"), "PATCH", body);
  if (code >= 200 && code < 300) Serial.println("[LIVE] Firebase update successful");
  else Serial.printf("[LIVE] Firebase update failed: %d\n", code);
}

void saveHistory() {
  if (authState != AUTH_READY || !validClock()) return;
  time_t now = time(nullptr);
  time_t slot = now - (now % HISTORY_SLOT_SECONDS);
  if (slot == lastHistorySlot) return;
  String body = multiPathJson("history", slot);
  if (body == "{}") return;
  int code = httpsRequest(dbUrl("/.json"), "PATCH", body);
  if (code >= 200 && code < 300) { lastHistorySlot = slot; Serial.printf("[HISTORY] Saved slot %lld\n", (long long)slot); }
  else Serial.printf("[HISTORY] Save failed: %d\n", code);
}

// Reads only the earliest three records before cutoff, then deletes just those
// keys. Numeric parsing, rather than lexicographic assumptions, guards against
// malformed/non-timestamp keys.
void cleanupHistory(uint32_t now) {
  if (!elapsed(now, lastCleanupMs, CLEANUP_INTERVAL_MS) || authState != AUTH_READY || !validClock()) return;
  lastCleanupMs = now;
  time_t cutoff = time(nullptr) - HISTORY_RETENTION_SECONDS;
  for (uint8_t i = 0; i < PZEM_COUNT; ++i) {
    if (!selected(i)) continue;
    String query = String("/history/pzem_") + ADDRESS[i] + ".json?orderBy=%22%24key%22&endAt=%22" + String((long long)cutoff) + "%22&limitToFirst=3&auth=" + idToken;
    String reply;
    int listCode = httpsRequest(String(FIREBASE_DATABASE_URL) + query, "GET", "", &reply);
    if (listCode < 200 || listCode >= 300) continue;
    int cursor = 0, removed = 0;
    while (removed < 3) {
      int quote = reply.indexOf('"', cursor); if (quote < 0) break;
      int end = reply.indexOf('"', quote + 1); if (end < 0) break;
      String key = reply.substring(quote + 1, end); cursor = end + 1;
      char *tail = nullptr; long long stamp = strtoll(key.c_str(), &tail, 10);
      if (!key.length() || *tail || stamp <= 0 || stamp >= cutoff) continue;
      int code = httpsRequest(dbUrl(String("/history/pzem_") + ADDRESS[i] + "/" + key + ".json"), "DELETE", "");
      if (code >= 200 && code < 300) removed++;
    }
    if (removed) Serial.printf("[CLEANUP] PZEM %u removed %d old record(s)\n", ADDRESS[i], removed);
  }
}

String apName() {
  return String("SmartEnergy-") + String((uint32_t)ESP.getEfuseMac(), HEX).substring(4);
}

void startPortal() {
  if (portalRunning) return;
  String ap = apName();
  wifiManager.setConfigPortalBlocking(false);
  wifiManager.setAPCallback([](WiFiManager *) { Serial.println("[WIFI] AP started; open http://192.168.4.1"); });
  wifiManager.startConfigPortal(ap.c_str());
  portalRunning = true;
  Serial.printf("[WIFI] Provisioning AP: %s\n", ap.c_str());
}

void serviceWiFi(uint32_t now) {
  wifiManager.process();
  if (WiFi.status() == WL_CONNECTED) {
    if (portalRunning) { portalRunning = false; Serial.printf("[WIFI] Connected: %s\n", WiFi.localIP().toString().c_str()); }
    return;
  }
  if (portalRunning) return;
  if (!elapsed(now, lastWiFiRetryMs, WIFI_RETRY_INTERVAL_MS)) return;
  lastWiFiRetryMs = now;
  startPortal();
}

void setup() {
  Serial.begin(SERIAL_MONITOR_BAUD);
  Serial.println("\n[BOOT] Smart Energy Monitoring System");
  if (PZEM_TEST_MODE && (PZEM_TEST_ADDRESS < 1 || PZEM_TEST_ADDRESS > 9)) { Serial.println("[BOOT] Invalid PZEM_TEST_ADDRESS"); while (true) delay(1000); }
  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  pinMode(MUX_S3, OUTPUT);
  selectPzemMux(0); // deterministic mux state (channel 0 / PZEM1) before Serial2 starts
  Serial2.begin(PZEM_BAUD, SERIAL_8N1, ESP32_RX_PIN, ESP32_TX_PIN);
  for (uint8_t i = 0; i < PZEM_COUNT; ++i) pzems[i] = PZEM004Tv30(Serial2, ESP32_RX_PIN, ESP32_TX_PIN, ADDRESS[i]);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  bool connected = false;
  if (strlen(WIFI_SSID)) {
    Serial.println("[WIFI] Trying optional WIFI_SSID from config.h");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    const uint32_t started = millis();
    while (WiFi.status() != WL_CONNECTED && (uint32_t)(millis() - started) < 10000UL) delay(200);
    connected = WiFi.status() == WL_CONNECTED;
  }
  if (!connected) {
    wifiManager.setConfigPortalBlocking(false);
    wifiManager.setAPCallback([](WiFiManager *) { Serial.println("[WIFI] AP started; open http://192.168.4.1"); });
    connected = wifiManager.autoConnect(apName().c_str());
  }
  if (connected) Serial.printf("[WIFI] Connected: %s\n", WiFi.localIP().toString().c_str());
  else { portalRunning = true; Serial.printf("[WIFI] Provisioning AP: %s (open http://192.168.4.1)\n", apName().c_str()); }
  if (PZEM_TEST_MODE) {
    Serial.printf("[BOOT] PZEM test mode: ON — monitoring ONLY address %u. This is a diagnostic build, not production.\n", PZEM_TEST_ADDRESS);
  } else {
    Serial.printf("[BOOT] PZEM test mode: OFF — monitoring all %u addresses (1..%u).\n", PZEM_COUNT, PZEM_COUNT);
  }
}

void loop() {
  uint32_t now = millis();
  serviceWiFi(now);
  if (WiFi.status() == WL_CONNECTED) syncClock();
  serviceAuthentication(now);
  pollMeters(now);
  uploadLive(now);
  saveHistory();
  cleanupHistory(now);
  delay(2);
}
