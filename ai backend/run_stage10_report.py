"""
run_stage10_report.py
----------------------
STAGE 10 real-Firebase verification script — runs the FULL existing AI
execution flow and appends AI bill prediction:

    Stage 1  load history (ai.data_loader)
    Stage 2  preprocess (ai.preprocessing)
    Stage 3  anomaly detection
    Stage 4/5 persist anomalies + faults
    Stage 7  peak-load detection
    Stage 8  maintenance risk
    Stage 9  forecasting + persistence (/ai/forecast)
    Stage 10 bill prediction + persistence (/ai/bill_prediction)  <-- this stage

Stage 10 is purely ADDITIVE: it consumes the SAME in-memory Stage 2 results
and the Stage 9 system forecast (no second data load) and writes only to
/ai/bill_prediction/<anchor-ts>, never touching any earlier namespace.

Optional env:
    BILL_RATE_PER_KWH   flat tariff in ₹/kWh used for the predicted-bill
                        number (defaults to 0.0 -> energies only, no bill).

Usage:
    python run_stage10_report.py
"""

from __future__ import annotations

import os
import sys

from ai import anomaly_detection, preprocessing, persist_ai_results
from ai import forecast
from ai import peak_detection
from ai import maintenance_risk
from ai import bill_prediction
from ai.config import get_settings


def main() -> int:
    try:
        settings = get_settings()
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    rate = float(os.environ.get("BILL_RATE_PER_KWH", "0.0") or "0.0")

    print(f"Discovering and preprocessing PZEM 1..{settings.pzem_count} from real Firebase data...\n")
    preprocess_results = preprocessing.run_preprocessing_pipeline(settings=settings)

    print("Running Stage 3 anomaly detection (operating-state-aware)...\n")
    anomaly_results = anomaly_detection.run_anomaly_detection_pipeline(
        settings=settings, preprocess_results=preprocess_results,
    )

    print("Persisting AI results to Firebase (Stages 4-5)...\n")
    stage5 = persist_ai_results.run_stage_5_pipeline(
        preprocess_results=preprocess_results, anomaly_results=anomaly_results,
    )
    print(f"Stage 5 anomaly writes: {stage5['anomalies']}")
    print(f"Stage 5 fault writes:   {stage5['faults']}\n")

    print("Running Stage 7 peak-load detection...\n")
    peak_detection.run_peak_detection_pipeline(
        settings=settings, preprocess_results=preprocess_results
    )

    print("Running Stage 8 maintenance risk...\n")
    maintenance_risk.run_maintenance_risk_pipeline(
        settings=settings, preprocess_results=preprocess_results
    )

    print("Running Stage 9 forecasting...\n")
    stage9 = forecast.run_stage_9_pipeline(
        settings=settings, preprocess_results=preprocess_results
    )
    print(f"Stage 9 per-PZEM forecast writes: {stage9['per_pzem']}")
    print(f"Stage 9 system forecast write:    {stage9['system']}\n")

    print("Running Stage 10 AI bill prediction...\n")
    actual_kwh = bill_prediction.compute_actual_energy_from_history(settings=settings)
    pred = bill_prediction.predict_bill_from_record(
        actual_kwh, stage9["system_result"], horizon="forecast_24h", rate=rate,
        billing_period="30d",
    )
    if pred.get("status") == "OK" and pred.get("anchor_timestamp"):
        written = bill_prediction.write_bill_prediction(pred, pred["anchor_timestamp"])
    else:
        written = False
        print(f"  Bill prediction withheld: {pred.get('reason')}")

    print("=" * 70)
    print("STAGE 10 BILL PREDICTION REPORT")
    print("=" * 70)
    if pred.get("status") == "OK":
        print(f"  Actual energy (so far)     : {pred['actual_energy_kwh']} kWh")
        print(f"  Forecasted energy ({ '24h' }) : {pred['forecast_energy_kwh']} kWh")
        print(f"  Estimated total           : {pred['estimated_total_energy_kwh']} kWh")
        print(f"  Rate                      : ₹{pred['rate']} / kWh")
        print(f"  Predicted bill            : ₹{pred['estimated_bill']}")
        print(f"  Difference vs actual      : ₹{pred['predicted_difference']}")
        print(f"  Forecast confidence       : {pred['forecast_confidence']}")
    else:
        print(f"  INSUFFICIENT: {pred.get('reason')}")
    print(f"  Persisted to /ai/bill_prediction : {written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
