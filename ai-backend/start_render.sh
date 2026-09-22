#!/bin/sh
set -e

# Write Firebase service account JSON from Render Secret to expected path
if [ -n "$FIREBASE_SERVICE_ACCOUNT_JSON" ]; then
  mkdir -p ./secrets
  printf '%s' "$FIREBASE_SERVICE_ACCOUNT_JSON" > ./secrets/firebase-service-account.json
fi

# Verify service account JSON exists before starting
if [ ! -f ./secrets/firebase-service-account.json ]; then
  echo "ERROR: FIREBASE_SERVICE_ACCOUNT_JSON secret not set. Service account JSON file missing." >&2
  exit 1
fi

# Create cache directory
mkdir -p ./.cache

# Start the API server with gunicorn
exec gunicorn -b 0.0.0.0:${API_PORT:-8000} wsgi:app
