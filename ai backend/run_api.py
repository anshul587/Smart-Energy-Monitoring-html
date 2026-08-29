"""
run_api.py
----------
Stage 15 entry point: start the read-only REST API for existing AI / energy
results.

    python run_api.py                 # http://127.0.0.1:8000
    API_PORT=8080 python run_api.py  # custom port

No authentication is configured. Local/development use only. See the security
notes in /api/v1/openapi.json before exposing this service.

Gunicorn:  gunicorn run_api:app
"""

from __future__ import annotations

import os

from ai.api_server import create_app

app = create_app()


def main() -> int:
    app.run(
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", "8000")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
