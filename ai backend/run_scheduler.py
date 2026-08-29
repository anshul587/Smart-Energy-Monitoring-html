"""
run_scheduler.py
----------------
STAGE 14 entry point: start the storage-efficient scheduled AI runner.

Runs the EXISTING Stage 1-13 pipeline on a schedule (incremental, idempotent
persistence) and generates the monthly PDF report. No new AI logic.

Usage:
    python run_scheduler.py                 # run forever (until SIGINT/SIGTERM)
    python run_scheduler.py --once         # run a single tick and exit
    python run_scheduler.py --once --dry   # single tick with a no-op runner

Configuration is via environment variables (see get_scheduler_config):
    SCHEDULER_ENABLED, AI_PROCESS_INTERVAL_MIN, MONTHLY_REPORT_HOUR,
    MONTHLY_REPORT_ALLOW_CURRENT, RETRY_COUNT, RETRY_DELAY_SEC,
    LOCK_TIMEOUT_SEC, SCHEDULER_LOG_LEVEL, SCHEDULER_STATE_FILE.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 14 scheduled AI runner")
    p.add_argument("--once", action="store_true", help="run a single tick and exit")
    p.add_argument("--dry", action="store_true", help="use a no-op runner (no Firebase)")
    args = p.parse_args()

    from ai.scheduler import Scheduler, get_scheduler_config, run_ai_pipeline, run_monthly_report_job

    cfg = get_scheduler_config()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.dry:
        sched = Scheduler(
            config=cfg,
            ai_runner=lambda settings, now: {"status": "dry"},
            monthly_runner=lambda settings, now, y, m: {"status": "dry"},
            has_new_data=lambda settings, ts, now: True,
        )
    else:
        sched = Scheduler(config=cfg)

    if args.once:
        logging.getLogger("ai.scheduler").info("Running single tick (--once)")
        sched.tick()
        sched._shutdown()
        return 0

    try:
        sched.run_forever(poll_seconds=60)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
