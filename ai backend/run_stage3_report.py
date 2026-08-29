"""
run_stage3_report.py
---------------------
STAGE 3 real-Firebase verification script.

Loads REAL history via Stage 1 (ai.data_loader), runs REAL Stage 2
preprocessing (ai.preprocessing), then REAL Stage 3 anomaly detection
(ai.anomaly_detection) — against every configured PZEM (1..PZEM_COUNT),
discovered dynamically. Nothing here is synthetic or fabricated: a
meter with too little (or no) ACTIVE historical data is reported as
INSUFFICIENT_DATA, not silently skipped, padded with fake rows, or
called a fault. No PZEM numbers and no classroom schedule are
hard-coded anywhere in this script — every meter 1..PZEM_COUNT is
queried fresh each run, and "active" is whatever Stage 3's
operating-state detector finds in that meter's own electrical readings.

Usage:
    python run_stage3_report.py
"""

from __future__ import annotations

import sys

from ai import anomaly_detection, preprocessing, persist_ai_results
from ai.config import get_settings


def _combined_pzem_report(pre, anom) -> str:
    """One PZEM's full Stage 1 -> Stage 2 -> Stage 3 picture: raw
    records, valid records, detected active/inactive observations,
    training observations, model status, and either the latest scored
    anomaly result or the reason training didn't happen."""
    lines = []
    lines.append(f"PZEM {pre.pzem_number}")
    lines.append(f"  Raw records (Stage 1, pre-cleaning):     {pre.record_count}")
    lines.append(f"  Valid records (Stage 2, post-cleaning):  {pre.valid_rows}")
    lines.append(f"  Invalid rows:                            {pre.invalid_rows}")
    lines.append(f"  Duplicates removed:                      {pre.duplicates_removed}")
    lines.append(f"  Missing values:                          {pre.missing_values}")
    lines.append(f"  Available days (of usable data):         {pre.available_days}")
    lines.append(f"  Stage 2 status:                          {pre.status}")
    if pre.status != "READY":
        lines.append(f"  Stage 2 reason:                          {pre.reason}")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"  Operating-state detection method:        {anom.operating_state_method}")
    lines.append(f"  Detected ACTIVE observations:             {anom.active_rows}")
    lines.append(f"  Detected INACTIVE observations:           {anom.inactive_rows}")
    lines.append(f"  Active days represented (qualifying):    {anom.active_days_represented}")
    lines.append(f"  Training observations (ACTIVE, clean):   {anom.training_rows}")
    lines.append(f"  Stage 3 model status:                    {anom.model_status}")

    if anom.model_status != "READY":
        lines.append(f"  Reason model was not trained:            {anom.reason}")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"  Features used:                           {', '.join(anom.features_used)}")
    lines.append(f"  Contamination / random_state:            {anom.contamination} / {anom.random_state}")

    scored = anom.result_frame[anom.result_frame["anomaly_label"] != "NOT_SCORED"]
    if scored.empty:
        lines.append("  No scored observations yet (most recent data is INACTIVE).")
    else:
        latest = scored.iloc[-1]
        lines.append(f"  Latest scored anomaly result:            {latest['anomaly_label']}")
        lines.append(f"  Anomaly score (raw decision_function):   {latest['anomaly_score']:.6f}")
        lines.append(f"  Anomaly score (normalized, this meter):  {latest['anomaly_score_normalized']:.4f}")
        lines.append(f"  Anomaly severity (provisional):          {latest['anomaly_severity_provisional']}")
        lines.append(
            "  NOTE: this is a statistical-deviation flag relative to this "
            "meter's own ACTIVE-operation history — NOT a fault diagnosis."
        )

    if anom.debug_traceback:
        lines.append(
            "  (!) An exception occurred while processing this meter — "
            "see full traceback output below."
        )
    lines.append("")
    return "\n".join(lines)


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

    print("Persisting AI results to Firebase (Stage 5)...\n")
    stage5 = persist_ai_results.run_stage_5_pipeline(
        preprocess_results=preprocess_results,
        anomaly_results=anomaly_results,
    )
    print(f"Stage 5 anomaly writes: {stage5['anomalies']}")
    print(f"Stage 5 fault writes:   {stage5['faults']}\n")

    print("=" * 70)
    print("PER-PZEM REPORT (Stage 1 -> Stage 2 -> Stage 3)")
    print("=" * 70)
    for n in sorted(preprocess_results):
        print(_combined_pzem_report(preprocess_results[n], anomaly_results[n]))

    summary = anomaly_detection.summarize_fleet(anomaly_results)
    print("=" * 50)
    print("FLEET SUMMARY")
    print("=" * 50)
    print(f"PZEMs analyzed:            {summary.analyzed}")
    print(f"INSUFFICIENT_DATA:         {summary.insufficient_data}")
    print(f"Currently showing ANOMALY: {summary.anomalous_now}")
    print(f"Currently NORMAL:          {summary.normal_now}")

    any_debug = any(r.debug_traceback for r in anomaly_results.values())
    if any_debug:
        print(
            "\nOne or more meters hit an unexpected exception during this run "
            "(see log output above). Full tracebacks:"
        )
        anomaly_detection.print_debug_tracebacks(anomaly_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
