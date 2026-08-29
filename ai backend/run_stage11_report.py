"""
run_stage11_report.py
---------------------
STAGE 11 real-Firebase verification script — runs the FULL existing AI
execution flow and appends AI energy-saving suggestions:

    Stage 1  load history (ai.data_loader)
    Stage 2  preprocess (ai.preprocessing)
    Stage 7  peak-load detection
    Stage 8  maintenance risk
    Stage 9  forecasting
    Stage 11 energy-saving suggestions + persistence (/ai/energy_saving)  <-- this stage

Stage 11 is purely ADDITIVE: it consumes the SAME in-memory Stage 2 results
and the Stage 7/8/9 result objects (no second data load) and writes only to
/ai/energy_saving/<anchor-ts>, never touching any earlier namespace.

Optional env:
    BILL_RATE_PER_KWH   flat tariff in ₹/kWh used for ESTIMATED savings
                        (defaults to 0.0 -> energies only, no cost saving).

Usage:
    python run_stage11_report.py
"""

from __future__ import annotations

import os
import sys

from ai import preprocessing
from ai import peak_detection
from ai import maintenance_risk
from ai import forecast
from ai import energy_saving
from ai.config import get_settings


def main() -> int:
    try:
        settings = get_settings()
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    rate = float(os.environ.get("BILL_RATE_PER_KWH", "0.0") or "0.0")

    print(f"Discovering and preprocessing PZEM 1..{settings.pzem_count} from real Firebase data...\n")
    pre = preprocessing.run_preprocessing_pipeline(settings=settings)

    print("Running Stage 7 peak-load detection...\n")
    peaks, _system_peak = peak_detection.run_peak_detection_pipeline(
        settings=settings, preprocess_results=pre
    )

    print("Running Stage 8 maintenance risk...\n")
    risks, _system_risk = maintenance_risk.run_maintenance_risk_pipeline(
        settings=settings, preprocess_results=pre
    )

    print("Running Stage 9 forecasting...\n")
    forecasts, _system_fc = forecast.run_forecast_pipeline(
        settings=settings, preprocess_results=pre
    )

    print("Building Stage 11 evidence and generating recommendations...\n")
    meters = {}
    for n in range(1, settings.pzem_count + 1):
        pr = pre.get(n)
        meters[n] = energy_saving.MeterEvidence(
            pzem_number=n,
            feature_frame=(pr.feature_frame if pr and pr.feature_frame is not None else None),
            peak_result=peaks.get(n),
            risk_result=risks.get(n),
            forecast_result=forecasts.get(n),
        )

    result = energy_saving.run_stage_11_pipeline(meters, rate=rate, force=False)
    recs = result["recommendations"]

    print("=" * 70)
    print("STAGE 11 ENERGY-SAVING REPORT")
    print("=" * 70)
    print(f"  Anchor timestamp : {result['anchor_timestamp']}")
    print(f"  Rate             : ₹{rate} / kWh")
    print(f"  Recommendations  : {len(recs)}")
    if recs:
        for r in recs:
            who = "SYSTEM" if r.pzem_number is None else f"PZEM {r.pzem_number}"
            sav = ""
            if r.potential_saving_kwh is not None:
                sav = f"  est. saving ~{r.potential_saving_kwh} kWh"
                if r.potential_cost_saving is not None:
                    sav += f" (₹{r.potential_cost_saving})"
            print(f"  [{r.priority}] {who} {r.recommendation_type}{sav}")
            print(f"         {r.reason}")
    else:
        print("  NO_RECOMMENDATION: insufficient evidence for any saving suggestion.")
    print(f"  Persisted to /ai/energy_saving : {result['persist']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
