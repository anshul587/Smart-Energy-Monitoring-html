#!/bin/sh
set -e

# Write Firebase service account JSON from Render Secret to expected path
if [ -n "$FIREBASE_SERVICE_ACCOUNT_JSON" ]; then
  mkdir -p ./secrets
  printf '%s' "$FIREBASE_SERVICE_ACCOUNT_JSON" > ./secrets/firebase-service-account.json
fi

# Create cache directory
mkdir -p ./.cache

# Start the API server with gunicorn
exec gunicorn -b 0.0.0.0:${API_PORT:-8000} wsgi:app
