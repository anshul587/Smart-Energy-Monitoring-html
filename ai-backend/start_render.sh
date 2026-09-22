#!/bin/sh
set -e

# Write Firebase service account JSON from Render Secret to expected path
# (legacy inline-secret flow; only used when the Secret File mount below is
# not configured, so it never overwrites the mounted secret).
if [ -n "$FIREBASE_SERVICE_ACCOUNT_JSON" ] && [ -z "$FIREBASE_SERVICE_ACCOUNT_PATH" ]; then
  mkdir -p ./secrets
  printf '%s' "$FIREBASE_SERVICE_ACCOUNT_JSON" > ./secrets/firebase-service-account.json
fi

# Verify the configured service-account file exists and is readable before
# starting. FIREBASE_SERVICE_ACCOUNT_PATH is the Render Secret File mount
# (or a path from .env on a dev machine); the JSON flow above produces the
# same default path when used.
SA_FILE="${FIREBASE_SERVICE_ACCOUNT_PATH:-./secrets/firebase-service-account.json}"
if [ ! -f "$SA_FILE" ] || [ ! -r "$SA_FILE" ]; then
  echo "ERROR: Firebase service account file not found or unreadable at $SA_FILE (set FIREBASE_SERVICE_ACCOUNT_PATH)." >&2
  exit 1
fi

# Create cache directory
mkdir -p ./.cache

# Start the API server with gunicorn
exec gunicorn -b 0.0.0.0:${API_PORT:-8000} wsgi:app
