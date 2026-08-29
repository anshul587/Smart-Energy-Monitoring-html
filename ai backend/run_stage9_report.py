"""
run_stage9_report.py
--------------------
STAGE 9 real-Firebase verification script — runs the FULL existing AI
execution flow and appends forecasting:

    Stage 1  load history (ai.data_loader, incremental cache)
    Stage 2  preprocess (ai.preprocessing)
    Stage 3  anomaly detection (ai.anomaly_detection)
    Stage 4  fault diagnosis (ai.fault_diagnosis, via Stage 5)
    Stage 5  persist anomalies + faults (ai.persist_ai_results -> /ai/*)
    Stage 7  peak-load detection (ai.peak_detection -> /ai/peaks/*)
    Stage 8  maintenance risk (ai.maintenance_risk -> /ai/maintenance/*)
    Stage 9  forecasting + persistence (this stage -> /ai/forecast/*)

Stage 9 is purely ADDITIVE: it consumes the same in-memory Stage 2 results
(no second data load) and writes only to /ai/forecast/pzem_N/<ts> and
/ai/forecast/system/<ts>, never touching /meters, /history, /alerts, or any
earlier /ai/* namespace.

Usage:
    python run_stage9_report.py
"""

from __future__ import annotations

import sys

from ai import anomaly_detection, preprocessing, persist_ai_results
from ai import forecast
from ai import peak_detection
from ai.config import get_settings


def main() -> int:
    try:
        settings = get_settings()
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    print(f"Discovering and preprocessing PZEM 1..{settings.pzem_count} from real Firebase data...\n")
    preprocess_results = preprocessing.run_preprocessing_pipeline(settings=settings)

    print("Running Stage 3 anomaly detection (operating-state-aware)...\n")
    anomaly_results = anomaly_detection.run_anomaly_detection_pipeline(
        settings=settings,
        preprocess_results=preprocess_results,
    )

    print("Persisting AI results to Firebase (Stages 4-5)...\n")
    stage5 = persist_ai_results.run_stage_5_pipeline(
        preprocess_results=preprocess_results,
        anomaly_results=anomaly_results,
    )
    print(f"Stage 5 anomaly writes: {stage5['anomalies']}")
    print(f"Stage 5 fault writes:   {stage5['faults']}\n")

    print("Running Stage 7 peak-load detection (reusing cached Stage 1-2 data)...\n")
    peaks, system_peak = peak_detection.run_peak_detection_pipeline(
        settings=settings, preprocess_results=preprocess_results
    )

    print("Running Stage 9 forecasting (reusing cached Stage 1-2 data)...\n")
    stage9 = forecast.run_stage_9_pipeline(
        settings=settings, preprocess_results=preprocess_results
    )
    print(f"Stage 9 per-PZEM forecast writes: {stage9['per_pzem']}")
    print(f"Stage 9 system forecast write:    {stage9['system']}\n")

    print("=" * 70)
    print("STAGE 9 FORECAST REPORT")
    print("(5-minute historical data only; explainable daily-profile model)")
    print("=" * 70)
    print(forecast.format_report(stage9["results"], stage9["system_result"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
