# SmartEnergyMonitor (ESP32 DevKit V1)

This project writes only to the existing dashboard contract:

- `meters/pzem_1` through `meters/pzem_9`
- `history/pzem_1/<Unix epoch seconds>` through `history/pzem_9/<Unix epoch seconds>`

It does not host or modify a dashboard.

## Critical UART hardware finding

Do **not** use the originally proposed diode direction (`PZEM TX -> anode; cathode -> common ESP32 RX`). UART is idle-high and communicates a start bit by driving LOW, which that direction blocks. It can also expose GPIO16 to an unsafe logic-high voltage.

The safe first test is exactly one PZEM, directly wired, with `PZEM_TEST_MODE true`. Before using all nine, validate a 5 V-to-3.3 V receive circuit: one open-collector/open-drain isolator per PZEM TX, a 3.3 V pull-up on the ESP32 RX bus, and level shifting for ESP32 TX if the PZEM input requires 5 V. Do not tie push-pull PZEM TX lines together.

## Arduino IDE setup

1. Install **ESP32 by Espressif Systems** in Boards Manager (current stable release).
2. Select **DOIT ESP32 DEVKIT V1**, select the correct serial port, and use 115200 baud Serial Monitor.
3. In Library Manager install:
   - **PZEM004Tv30** by Jakub Mandula — addressed PZEM-004T V3/V4 Modbus-style reads.
   - **WiFiManager** by tzapu — persistent Wi-Fi provisioning portal, DNS, scanning and credential storage.
4. Open `SmartEnergyMonitor.ino`, edit `config.h`, and set `FIREBASE_DEVICE_PASSWORD` locally. Never commit it.

## First test

1. Keep `PZEM_TEST_MODE true` and set `PZEM_TEST_ADDRESS` to the meter’s existing address, 1–9.
2. Wire one powered PZEM directly: PZEM TX -> GPIO16, PZEM RX -> GPIO17, common ground. Confirm logic voltage safety before connecting any 5 V UART TX directly to GPIO16.
3. Upload. If no saved Wi-Fi exists, join the `SmartEnergy-XXXX` access point and open `http://192.168.4.1`; the portal saves credentials in NVS.
4. Confirm `[PZEM N] OK`, `[FIREBASE] Authentication successful`, and `[LIVE] Firebase update successful` in Serial Monitor.
5. Check `meters/pzem_N` in RTDB. A valid real-world clock is required before the first five-minute history write.

## Multi-PZEM test and final mode

After the bus circuit is validated, set `PZEM_TEST_MODE false`. The same sketch polls addresses 1–9 sequentially. Test two meters first; verify each path gets readings from the correct device before installing all nine.

## Timing and recovery

- PZEM poll: 2.5 s
- Live RTDB PATCH: 10 s
- History: current NTP-backed five-minute Unix slot; re-writing a slot is idempotent.
- Wi-Fi reconnect: 30 s
- Firebase email/password retry: 15 s, exponential backoff to 15 min.
- Token refresh: five minutes before expiry, governed by the same retry timer.
- Cleanup: every six hours, reads and deletes at most three old records per active meter. It compares parsed numeric Unix timestamps and never uses `millis()` as data time.

PZEM polling continues when Wi-Fi/Firebase is unavailable. Authentication is scheduled rather than called from every loop iteration.

## Firebase setup and security rules

Enable **Email/Password** in Firebase Authentication and create the configured device user. RTDB rules must permit that authenticated user to read/write only the required `meters` and `history` paths. Do not leave production RTDB rules publicly writable.

The sketch presently uses `WiFiClientSecure::setInsecure()` only for bring-up compatibility with Arduino-ESP32 installs that lack a configured CA bundle. Before production, replace it with a maintained Google CA/certificate bundle and test all three Firebase HTTPS hosts (Identity Toolkit, Secure Token, RTDB). Do not deploy device credentials with TLS validation disabled.

## Status

- Code complete: yes
- Compiled in this environment: not verified (Arduino ESP32 toolchain/hardware unavailable)
- Logic verified by review: yes
- Single-PZEM hardware test: required
- Nine-PZEM electrical bus: hardware correction/validation required
