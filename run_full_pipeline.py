#!/usr/bin/env python3
"""Run the full 30-day AI/ML pipeline with synthetic data."""
import os
import sys

# Set DATA_SOURCE to synthetic before any AI imports
os.environ["DATA_SOURCE"] = "synthetic"

# Load .env manually (for LLM config etc.)
env_path = "ai backend/.env"
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

sys.path.insert(0, "ai backend")

from ai.config import get_settings
from ai import preprocessing, anomaly_detection, fault_diagnosis, peak_detection
from ai import forecast, bill_prediction, energy_saving, maintenance_risk
from ai import persist_ai_results

print("=" * 60)
print("FULL 30-DAY AI/ML PIPELINE - SYNTHETIC DATA")
print("=" * 60)

settings = get_settings()
print(f"\nSettings:")
print(f"  pzem_count: {settings.pzem_count}")
print(f"  history_retention_days: {settings.history_retention_days}")
print(f"  data_source: {settings.data_source}")
print(f"  peak_power_threshold_w: {settings.peak_power_threshold_w}")

# ============================================================
# Step 1: Preprocessing
# ============================================================
print("\n" + "=" * 60)
print("STAGE 1: PREPROCESSING")
print("=" * 60)

pre_results = preprocessing.run_preprocessing_pipeline(settings=settings)
print(f"  Preprocessed {len(pre_results)} PZEM meters")

ready_count = sum(1 for r in pre_results.values() if r.status == "READY")
insuf_count = sum(1 for r in pre_results.values() if r.status == "INSUFFICIENT_DATA")
print(f"  READY: {ready_count}, INSUFFICIENT_DATA: {insuf_count}")

# Show per-meter status
for n in sorted(pre_results.keys()):
    r = pre_results[n]
    print(f"  PZEM {n}: status={r.status}, valid_rows={r.valid_rows}, "
          f"record_count={r.record_count}, available_days={r.available_days}")

# ============================================================
# Step 2: Anomaly Detection
# ============================================================
print("\n" + "=" * 60)
print("STAGE 2: ANOMALY DETECTION")
print("=" * 60)

anomaly_results = anomaly_detection.run_anomaly_detection_pipeline(
    settings=settings, preprocess_results=pre_results
)
print(f"  Anomaly-detected {len(anomaly_results)} PZEM meters")

anomaly_ready = sum(1 for r in anomaly_results.values() if r.model_status == "READY")
anomaly_insuf = sum(1 for r in anomaly_results.values() if r.model_status != "READY")
print(f"  READY: {anomaly_ready}, NOT READY: {anomaly_insuf}")

# Show per-meter anomaly status
for n in sorted(anomaly_results.keys()):
    ar = anomaly_results[n]
    status = ar.model_status
    reason = ar.reason or ""
    training_rows = ar.training_rows
    # Get latest anomaly label
    label = "N/A"
    if ar.result_frame is not None and not ar.result_frame.empty:
        scored = ar.result_frame[ar.result_frame["anomaly_label"] != "NOT_SCORED"]
        if not scored.empty:
            label = scored.iloc[-1]["anomaly_label"]
    print(f"  PZEM {n}: status={status}, training_rows={training_rows}, "
          f"latest_anomaly={label}, method={ar.operating_state_method}")

# ============================================================
# Step 3: Fault Diagnosis
# ============================================================
print("\n" + "=" * 60)
print("STAGE 3: FAULT DIAGNOSIS")
print("=" * 60)

fault_results = fault_diagnosis.run_fault_diagnosis_pipeline(pre_results)
print(f"  Fault-detected events across {len(fault_results)} PZEM meters")

total_events = 0
for n in sorted(fault_results.keys()):
    events = fault_results[n]
    total_events += len(events)
    emergency = [e for e in events if e.severity == "EMERGENCY"]
    warning = [e for e in events if e.severity == "WARNING"]
    comm_degraded = [e for e in events if e.fault_type == "communication_degraded"]
    print(f"  PZEM {n}: total={len(events)}, emergency={len(emergency)}, "
          f"warning={len(warning)}, comm_degraded={len(comm_degraded)}")

# ============================================================
# Step 4: Peak Detection
# ============================================================
print("\n" + "=" * 60)
print("STAGE 4: PEAK DETECTION")
print("=" * 60)

peak_pair = peak_detection.run_peak_detection_pipeline(settings=settings, preprocess_results=pre_results)
peaks, system_peak = peak_pair

print(f"  Per-PZEM peak results: {len(peaks)} meters")
found_count = sum(1 for r in peaks.values() if r.status == "PEAK_FOUND")
no_peak_count = sum(1 for r in peaks.values() if r.status == "NO_PEAK")
print(f"  PEAK_FOUND: {found_count}, NO_PEAK: {no_peak_count}")

# Show per-meter peak status
for n in sorted(peaks.keys()):
    pr = peaks[n]
    if pr.status == "PEAK_FOUND":
        print(f"  PZEM {n}: PEAK_FOUND at {pr.peak_power_w}W, "
              f"duration={pr.peak_duration_seconds}s, sustained={pr.sustained}")
    else:
        print(f"  PZEM {n}: NO_PEAK - {pr.reason}")

if system_peak and system_peak.status == "PEAK_FOUND":
    print(f"  SYSTEM PEAK: {system_peak.total_peak_power_w}W, "
          f"meters={system_peak.meters_analyzed}, dominant={system_peak.dominant_pzems}")
else:
    print(f"  SYSTEM: NO_PEAK - {system_peak.reason if system_peak else 'N/A'}")

# ============================================================
# Step 5: Forecasting
# ============================================================
print("\n" + "=" * 60)
print("STAGE 5: POWER FORECASTING")
print("=" * 60)

fc_pair = forecast.run_forecast_pipeline(settings=settings, preprocess_results=pre_results)
meter_results, system_forecast = fc_pair

print(f"  Per-PZEM forecast results: {len(meter_results)} meters")
forecast_ready = sum(1 for r in meter_results.values() if r.status == "FORECAST")
no_forecast = sum(1 for r in meter_results.values() if r.status == "NO_FORECAST")
print(f"  FORECAST: {forecast_ready}, NO_FORECAST: {no_forecast}")

# Show per-meter forecast status
for n in sorted(meter_results.keys()):
    fr = meter_results[n]
    if fr.status == "FORECAST":
        print(f"  PZEM {n}: 24h forecast={fr.forecast_24h.get('count') if fr.forecast_24h else 'N/A'} points, "
              f"7d forecast={fr.forecast_7d.get('count') if fr.forecast_7d else 'N/A'} points, "
              f"confidence={fr.forecast_24h.get('confidence') if fr.forecast_24h else 'N/A'}")
    else:
        print(f"  PZEM {n}: NO_FORECAST - {fr.reason}")

if system_forecast and system_forecast.status == "FORECAST":
    print(f"  SYSTEM: 24h={system_forecast.forecast_24h.get('count')} points, "
          f"7d={system_forecast.forecast_7d.get('count')} points, "
          f"confidence={system_forecast.forecast_24h.get('confidence')}")
else:
    print(f"  SYSTEM: NO_FORECAST - {system_forecast.reason if system_forecast else 'N/A'}")

# ============================================================
# Step 6: Bill Prediction
# ============================================================
print("\n" + "=" * 60)
print("STAGE 6: BILL PREDICTION")
print("=" * 60)

# Compute actual energy from history
actual_energy = bill_prediction.compute_actual_energy_from_history(settings=settings)
print(f"  Actual energy from history: {actual_energy} kWh")

# Run bill prediction with the system forecast
bill_result = bill_prediction.predict_bill_from_record(
    actual_energy, system_forecast, horizon="forecast_24h", rate=0.0, billing_period="30d"
)
print(f"  Bill prediction status: {bill_result.get('status')}")
if bill_result.get('status') == 'OK':
    print(f"  Estimated bill: {bill_result.get('estimated_bill')}")
    print(f"  Actual energy: {bill_result.get('actual_energy_kwh')} kWh")
    print(f"  Forecast energy: {bill_result.get('forecast_energy_kwh')} kWh")
    print(f"  Estimated total energy: {bill_result.get('estimated_total_energy_kwh')} kWh")
else:
    print(f"  Reason: {bill_result.get('reason')}")

# Persist bill prediction
if bill_result.get('status') == 'OK' and bill_result.get('anchor_timestamp'):
    written = bill_prediction.write_bill_prediction(bill_result, bill_result['anchor_timestamp'])
    print(f"  Bill prediction persisted: {written}")
else:
    print(f"  Bill prediction not persisted (no anchor timestamp or not OK)")

# ============================================================
# Step 7: Energy Saving
# ============================================================
print("\n" + "=" * 60)
print("STAGE 7: ENERGY SAVING SUGGESTIONS")
print("=" * 60)

# Build meter evidence from previous results
meters = {}
for n in range(1, settings.pzem_count + 1):
    prr = pre_results.get(n)
    meters[n] = energy_saving.MeterEvidence(
        n,
        feature_frame=(prr.feature_frame if prr and prr.feature_frame is not None else None),
        peak_result=(peaks.get(n) if peaks else None),
        risk_result=(maintenance_risk.run_maintenance_risk_pipeline(
            settings=settings, preprocess_results={n: pre_results.get(n)})[0].get(n) if pre_results.get(n) else None),
        forecast_result=(meter_results.get(n) if meter_results else None),
    )

energy_saving_res = energy_saving.run_stage_11_pipeline(meters, rate=0.0, force=False)
recs = energy_saving_res.get('recommendations', [])
print(f"  Generated {len(recs)} energy-saving recommendations")

# Show recommendations by priority
for r in recs:
    pzem = r.pzem_number if r.pzem_number else "SYSTEM"
    print(f"  [{r.priority}] PZEM {pzem}: {r.recommendation_type} - {r.reason[:80]}...")

# Persist energy saving
if bill_result.get('status') == 'OK' and bill_result.get('anchor_timestamp'):
    persist = energy_saving.write_energy_saving(recs, bill_result['anchor_timestamp'], rate=0.0, force=False)
    print(f"  Energy saving persisted: {persist}")
else:
    print(f"  Energy saving not persisted (no anchor timestamp)")

# ============================================================
# Step 8: Maintenance Risk
# ============================================================
print("\n" + "=" * 60)
print("STAGE 8: MAINTENANCE RISK")
print("=" * 60)

maint_pair = maintenance_risk.run_maintenance_risk_pipeline(
    settings=settings, preprocess_results=pre_results,
    anomaly_results=anomaly_results, fault_events=fault_results, peak_results=peaks
)
maint_results, system_summary = maint_pair

print(f"  Maintenance risk assessed for {maint_results.get('meters_analyzed', len(maint_results))} meters")
print(f"  System sufficiency: {system_summary.system_data_sufficiency}")

high_risk = system_summary.high_risk_meters
watch_meters = system_summary.watch_meters
normal_meters = system_summary.normal_meters
insuf_meters = system_summary.insufficient_data_meters

print(f"  HIGH risk: {high_risk}")
print(f"  WATCH: {watch_meters}")
print(f"  NORMAL: {normal_meters}")
print(f"  INSUFFICIENT DATA: {insuf_meters}")

if system_summary.highest_risk_pzem is not None:
    top = next(r for r in maint_results.values() if r.pzem_number == system_summary.highest_risk_pzem)
    print(f"  Highest risk: PZEM {top.pzem_number}, score={top.risk_score}, level={top.risk_level}")
    print(f"  Top indicators: {top.indicators[:3]}")

# Persist system maintenance risk
if system_summary.timestamp:
    written = maintenance_risk.write_system_summary(system_summary)
    print(f"  System maintenance risk persisted: {written}")
else:
    print(f"  System maintenance risk not persisted (no timestamp)")

# Per-PZEM persistence
for n in sorted(maint_results.keys()):
    r = maint_results[n]
    if r.status == 'RISK_ASSESSED' and r.risk_score is not None:
        written = maintenance_risk.write_risk_result(r)
        if written:
            print(f"  PZEM {n} risk persisted")

# ============================================================
# Step 9: Persist anomalies and faults (Stage 5)
# ============================================================
print("\n" + "=" * 60)
print("STAGE 5: PERSIST AI RESULTS (anomalies + faults)")
print("=" * 60)

stage5 = persist_ai_results.run_stage_5_pipeline(pre_results, anomaly_results)
anomaly_written = sum(stage5['anomalies'].values())
fault_written = sum(stage5['faults'].values())
print(f"  Anomaly results written: {anomaly_written} PZEMs")
print(f"  Fault results written: {fault_written} PZEMs")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"  Preprocessing: {ready_count}/{len(pre_results)} READY")
print(f"  Anomaly detection: {anomaly_ready}/{len(anomaly_results)} READY")
print(f"  Fault diagnosis: {total_events} total events across {len(fault_results)} meters")
print(f"  Peak detection: {found_count}/{len(peaks)} PEAK_FOUND")
print(f"  Forecasting: {forecast_ready}/{len(meter_results)} FORECAST")
print(f"  Bill prediction: {bill_result.get('status')}")
print(f"  Energy saving: {len(recs)} recommendations")
print(f"  Maintenance risk: {system_summary.system_data_sufficiency}")
print(f"  Stage 5 persistence: {anomaly_written} anomalies, {fault_written} faults")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)