# AI backend — Stage 1: Firebase historical-data loader

This is the first stage of the AI/ML layer for **Smart Industrial Energy
Monitoring**. It sits alongside — not inside — the existing ESP32 firmware
and dashboard, and only reads two paths that already exist in your
Firebase RTDB:

```
meters/pzem_1 .. meters/pzem_9          (live, unchanged, read-only here)
history/pzem_1/<unix-seconds> .. pzem_9 (5-min slots, ~60 days, read-only here)
```

Nothing in this stage writes to Firebase, modifies the ESP32 firmware, or
touches `index.html` / `script.js` / `style.css`. Those changes start in
later stages once there's real AI output to display.

## What's in this stage

```
ai-backend/
  ai/
    __init__.py
    config.py         # env-var settings, shared by every later stage
    data_loader.py     # THE deliverable: Firebase -> pandas, with caching
  tests/
    test_data_loader.py
  requirements.txt
  .env.example
  .gitignore
```

### `ai/config.py`
Loads all configuration from environment variables (`.env` locally, real
env vars in deployment) — see `.env.example`. Nothing is hardcoded, and
the Firebase credential used here (a service-account JSON) is intentionally
separate from the ESP32's device email/password in `config.h`.

### `ai/data_loader.py`
- `fetch_meter_history(n)` / `fetch_all_history()` — the real work:
  - **Incremental**: caches each meter's history to a local Parquet file
    and only requests the slots newer than the newest cached timestamp on
    subsequent calls, instead of re-downloading up to 60 days every run.
  - **Honest about data quality**: malformed rows (bad timestamp key,
    non-numeric field, legacy bare-number row, missing field) are dropped
    and counted, never guessed at. Duplicate timestamp keys collapse to
    the latest value.
  - **Honest about data quantity**: `available_days` reports the actual
    span of data found — if the system has only been running 12 days, it
    reports 12, not the configured 60-day target.
  - **Degrades gracefully**: if Firebase can't be reached, falls back to
    whatever's cached on disk and sets `served_from_cache_only=True`
    rather than crashing the whole pipeline. One meter's Firebase error
    doesn't block the other eight in `fetch_all_history()`.

## Setup

```bash
cd ai-backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: point FIREBASE_SERVICE_ACCOUNT_PATH at your service-account
# JSON (Firebase console -> Project settings -> Service accounts ->
# Generate new private key), and confirm FIREBASE_DATABASE_URL matches
# the one already in config.h / script.js.
```

## Testing

```bash
pytest tests/test_data_loader.py -v
```

The tests mock the Firebase Admin SDK entirely (no real project/network
needed) and specifically cover the cases called out in the project spec:
normal data, missing data, malformed data, duplicate keys, Firebase
unavailable (with fallback to cache), no historical data, insufficient
historical data (reports actual span), and one meter's failure not
blocking the other eight.

Since this sandbox has no network access, I validated the actual logic
here directly (swapping the Parquet cache for CSV only for that dry run,
since `pyarrow` couldn't be installed offline) — all cases passed
deterministically across repeated runs. The shipped tests use the real
Parquet-based cache from `requirements.txt` and should be run in your
environment with `pip install -r requirements.txt` first.

## A note on the cache format

Local caches are Parquet (`pyarrow` backend) — compact and fast for a
9-meter × 60-day × 5-minute-interval dataset (≈155k rows total at full
retention). If you'd rather avoid the `pyarrow` dependency, the cache
read/write is isolated to `_load_cache()` / `_save_cache()` in
`data_loader.py` and can be swapped for CSV with no changes elsewhere.

## What's next (Stage 2)

`ai/preprocessing.py` — turns each meter's raw `HistoryLoadResult.frame`
into the feature set the anomaly model needs: rolling averages/std,
deviation from baseline, time-of-day and day-of-week features, per the
project's Stage 3 anomaly-detection requirements. I'll build and test that
next, then move to Stage 3 (Isolation Forest).
