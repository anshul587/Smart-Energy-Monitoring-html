"""
run_stage13_report.py
----------------------
STAGE 13 entry point: build the MONTHLY energy report (PDF) from the existing
AI pipeline outputs. Daily reports have been removed; this script produces only
the monthly PDF.

It reuses the SAME Stage 2 feature frames and Stage 3/4/7/8/9/10/11 results that
every other stage already computes — no second data load. When no Firebase data
is available (or --demo), a deterministic demo input is used so the report
mechanism is always exercisable.

Usage:
    python run_stage13_report.py monthly
    python run_stage13_report.py monthly --demo
    python run_stage13_report.py monthly --year 2026 --month 7
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

from ai import report_generator as rg


def _build_input():
    """Build a real ReportInput from the live pipelines; fall back to demo."""
    try:
        from ai.config import get_settings
        settings = get_settings()
        return rg.build_report_input_from_pipelines(settings), False
    except Exception as exc:  # noqa: BLE001 - report must never hard-fail
        print(f"[stage13] real data unavailable ({exc}); using demo input.", file=sys.stderr)
        return rg.demo_input(), True


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 13 monthly energy report (PDF)")
    p.add_argument("kind", choices=["monthly"], help="only 'monthly' is supported")
    p.add_argument("--demo", action="store_true", help="force offline deterministic demo")
    p.add_argument("--year", type=int, help="monthly: year (default current)")
    p.add_argument("--month", type=int, help="monthly: month 1-12 (default current)")
    p.add_argument("--out", help="override output root (default Dashboard reports/)")
    args = p.parse_args()

    rate = float(os.environ.get("BILL_RATE_PER_KWH", "0.0") or "0.0")

    if args.demo:
        data = rg.demo_input()
        used_demo = True
    else:
        data, used_demo = _build_input()

    res = rg.generate_monthly_report(data=data, year=args.year, month=args.month,
                                    output_dir=args.out, rate=rate)

    period = f"{args.year or 'current'}-{args.month or 'current'}"
    print("=" * 70)
    print(f"STAGE 13 MONTHLY ENERGY REPORT  (period: {period})")
    print("=" * 70)
    print(f"  Source        : {'demo input' if used_demo else 'live pipeline data'}")
    print(f"  PDF report    : {res['pdf']}")
    print(f"  PZEM analyzed : {res['report']['total_pzem']}")
    print(f"  AI recs       : {len(res['report']['ai_insights']['recommendations'])}")
    print(f"  Alerts        : {len(res['report']['alerts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
