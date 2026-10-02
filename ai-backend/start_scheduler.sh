#!/bin/sh
set -e

# Stage 14 scheduler process. Mirrors start_render.sh's credential resolution
# EXACTLY so the worker authenticates to Firebase the same way the web API
# does, then runs the scheduler's own entry point instead of gunicorn.

# Write Firebase service account JSON from Render Secret to expected path
# (legacy inline-secret flow; only used when the Secret File mount below is
# not configured, so it never overwrites the mounted secret).
if [ -n "$FIREBASE_SERVICE_ACCOUNT_JSON" ] && [ -z "$FIREBASE_SERVICE_ACCOUNT_PATH" ]; then
  mkdir -p ./secrets
  printf '%s' "$FIREBASE_SERVICE_ACCOUNT_JSON" > ./secrets/firebase-service-account.json
fi

SA_FILE="${FIREBASE_SERVICE_ACCOUNT_PATH:-./secrets/firebase-service-account.json}"
if [ ! -f "$SA_FILE" ] || [ ! -r "$SA_FILE" ]; then
  echo "ERROR: Firebase service account file not found or unreadable at $SA_FILE (set FIREBASE_SERVICE_ACCOUNT_PATH)." >&2
  exit 1
fi

# Stage 1 caches per-meter history here and the scheduler's has_new_data()
# guard reads it back, so the directory must exist before the first tick.
mkdir -p ./.cache

exec python -m ai.scheduler