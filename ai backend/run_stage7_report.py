"""
run_stage7_report.py
--------------------
STAGE 7 real-Firebase verification script — runs the FULL existing AI
execution flow and appends peak-load detection:

    Stage 1  load history (ai.data_loader, incremental cache)
    Stage 2  preprocess (ai.preprocessing)
    Stage 3  anomaly detection (ai.anomaly_detection)
    Stage 4  fault diagnosis (ai.fault_diagnosis, via Stage 5)
    Stage 5  persist anomalies + faults (ai.persist_ai_results -> /ai/*)
    Stage 7  peak-load detection + persistence (this stage -> /ai/peaks/*)

Stage 7 is purely ADDITIVE: it consumes the same in-memory Stage 2
results (no second data load), writes only to /ai/peaks/pzem_N/<ts> and
/ai/peaks/system/<ts>, and never touches /meters, /history, /alerts,
/ai/anomalies or /ai/faults. Peaks are NOT faults/alerts/emergencies.

Usage:
    python run_stage7_report.py
"""

from __future__ import annotations

import sys

from ai import anomaly_detection, preprocessing, persist_ai_results
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
        settings=settings,
        preprocess_results=preprocess_results,
    )

    print("Persisting Stage 7 peaks to /ai/peaks...\n")
    per_pzem_writes = {n: 1 if peak_detection.write_peak_result(r) else 0
                       for n, r in sorted(peaks.items())}
    system_write = 1 if peak_detection.write_system_peak(system_peak) else 0
    print(f"Stage 7 per-PZEM peak writes: {per_pzem_writes}")
    print(f"Stage 7 system peak write:    {system_write}\n")

    print("=" * 70)
    print("STAGE 7 PEAK-LOAD REPORT")
    print("(5-minute historical data only; a peak is not a fault or alert)")
    print("=" * 70)
    print(peak_detection.format_report(peaks, system_peak))

    if settings.peak_power_threshold_w > 0:
        print(f"Configured peak annotation threshold: "
              f"{settings.peak_power_threshold_w} W (informational only)")
    else:
        print("No peak threshold configured (PEAK_POWER_THRESHOLD_W unset/0); "
              "peaks reported as observed maxima.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
