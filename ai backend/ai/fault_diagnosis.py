"""
ai/fault_diagnosis.py
---------------------
STAGE 4: Fault Diagnosis.

Diagnoses supported electrical fault types using actual PZEM data
that has already been preprocessed through Stage 2 and evaluated
by Stage 3 anomaly detection.

New fault types:
  - Overvoltage
  - Undervoltage
  - Overcurrent
  - Power-factor drop
  - Frequency deviation
  - Abnormally high power
  - PZEM communication/data-quality failure

Every diagnosed event contains:
  pzem_number, timestamp, fault_type, severity, measured_value,
  reason, evidence/confidence

Severity classification: NORMAL | WARNING | EMERGENCY
Only EMERGENCY triggers the buzzer and red-blink behavior.

Firebase alert structure with edge/state-change logic is defined here
but written by the caller (ESP32 firmware).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .config import get_settings
from .preprocessing import PreprocessResult, READING_FIELDS

# ---------------------------------------------------------------------------
# Fault diagnosis thresholds — explicit, documented, tunable via Settings
# ---------------------------------------------------------------------------

# Overvoltage: voltage > threshold
# Default: 250V (typical mains limit) — adjustable per PZEM or globally
FAULT_OVERVOLTAGE_V = 250.0

# Undervoltage: voltage < threshold
# Default: 180V (lower mains limit) — adjustable
FAULT_UNDERVOLTAGE_V = 180.0

# Overcurrent: current > threshold
# Default: 30A (adjust for your CT/pctransformer rating)
FAULT_OVERCURRENT_A = 30.0

# Power-factor drop: pf < threshold
# Default: 0.85 — a declining PF often indicates harmonic distortion
# or reactive-power problems
FAULT_PF_DROP = 0.85

# Frequency deviation: |freq - 50| > delta  (or |freq - 60| > delta for 60 Hz grids)
# Default: 2 Hz tolerance on 50 Hz grid
FAULT_FREQ_DEVIATION_HZ = 2.0

# Abnormally high power: power > threshold
# Default: 5000 W — above this is likely a heavy load or fault
FAULT_HIGH_POWER_W = 5000.0

# PZEM communication / data-quality failure
# Triggered when a meter has INSUFFICIENT_DATA or very few valid readings
# (fewer than 2 valid rows in the latest 5-min slot window)
FAULT_COMM_FAILURE = "communication_failure"


@dataclass
class FaultEvent:
    """A diagnosed fault event with full metadata.

    Attributes
    ----------
    pzem_number : int
        The PZEM meter number (1-indexed).
    timestamp : int
        Unix seconds of the reading when the fault was detected.
    fault_type : str
        One of the FAULT_* constants above.
    severity : str
        "NORMAL" | "WARNING" | "EMERGENCY".
    measured_value : float
        The reading value that triggered the fault.
    reason : str
        Human-readable explanation of why this fault was diagnosed.
    evidence : Optional[dict]
        Additional measured values supporting the diagnosis, e.g.
        {"voltage": 254.2, "current": 12.3, "pf": 0.71}.
    confidence : float = 1.0
        How certain we are the fault is real (0.0 .. 1.0).
    """

    pzem_number: int
    timestamp: int
    fault_type: str
    severity: str
    measured_value: float
    reason: str
    evidence: Optional[dict] = field(default=None, repr=False)
    confidence: float = 1.0

    def to_firebase_dict(self) -> dict:
        """Serialize for Firebase alert structure.

        Returns a dict with the fields that the ESP32 firmware writes
        to the "alerts" path. The caller must handle edge/state-change
        logic (fault inactive→active = new event, etc.).
        """
        base = {
            "pzem_number": self.pzem_number,
            "fault_type": self.fault_type,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "measured_value": self.measured_value,
            "reason": self.reason,
        }
        if self.evidence:
            base["evidence"] = self.evidence
        base["emergency"] = self.severity == "EMERGENCY"
        return base


# ---------------------------------------------------------------------------
# Per-meter fault diagnosis
# ---------------------------------------------------------------------------

def diagnose_meter_faults(
    pzem_number: int,
    preprocess_result: PreprocessResult,
    settings: Optional[object] = None,
) -> tuple[list[FaultEvent], list[FaultEvent]]:
    """Diagnose all supported fault types for ONE PZEM meter.

    Returns
    -------
    (emergency_events, warning_events)
        Each is a list of FaultEvent. Only EMERGENCY events should
        trigger the buzzer and red-blink. WARNING events are logged
        but do not trigger hardware actions.
    """
    settings = settings or get_settings()

    events: list[FaultEvent] = []
    emergency_events: list[FaultEvent] = []

    # If we don't have clean data, we may still flag a communication
    # failure — but only if there's insufficient data at all.
    # If the meter is READY with valid data, proceed with electrical
    # fault checks. If INSUFFICIENT_DATA, only flag comm failure.
    if preprocess_result.status != "READY":
        # No valid data to diagnose from — flag communication failure
        # only if the meter has essentially no data.
        record_count = preprocess_result.record_count
        if record_count == 0 or record_count < 3:
            ts = preprocess_result.newest_timestamp or 0
            events.append(
                FaultEvent(
                    pzem_number=pzem_number,
                    timestamp=ts,
                    fault_type=FAULT_COMM_FAILURE,
                    severity="EMERGENCY",
                    measured_value=0.0,
                    reason=f"No usable historical data available for PZEM {pzem_number} "
                           f"({record_count} record(s)); cannot diagnose electrical faults.",
                    evidence={"record_count": record_count},
                    confidence=0.8,
                )
            )
            emergency_events.append(events[0])
            return emergency_events, []

        # Even with some data, if we can't compute features, stop here.
        # (The caller should have already checked status == READY.)
        return emergency_events, []

    # At this point we have a READY meter with valid data.
    # The feature frame may be None if valid_rows < MIN_VALID_ROWS but
    # that case should have been caught above. If somehow we have a
    # feature frame, use it; otherwise we only have the clean frame
    # metadata. For fault diagnosis we primarily need the raw readings,
    # which are available via the data loader — but the PreprocessResult
    # only carries metadata. We will work with what we have and flag
    # faults based on the most recent available readings.

    # Build a mapping of the latest readings from the feature frame if
    # available, otherwise we fall back to what the PreprocessResult
    # tells us (oldest/newest timestamps etc.). In a fully wired system
    # the ESP32 would send the latest readings, but here we diagnose
    # based on the preprocessed data that is already in memory.

    feature_frame = preprocess_result.feature_frame
    if feature_frame is not None and not feature_frame.empty:
        # Use the most recent row for fault diagnosis
        latest = feature_frame.iloc[-1]
        # Extract the latest readings per field
        latest_readings = {}
        for field in READING_FIELDS:
            if field in latest.index:
                val = float(latest[field])
                # Only include non-NaN values
                if np.isfinite(val):
                    latest_readings[field] = val
    else:
        latest_readings = {}

    # ---- Electrical fault diagnostics (only when we have valid readings) ----

    # 1. Overvoltage
    voltage = latest_readings.get("voltage")
    if voltage is not None and voltage > FAULT_OVERVOLTAGE_V:
        events.append(
            FaultEvent(
                pzem_number=pzem_number,
                timestamp=int(latest["timestamp"]),
                fault_type="overvoltage",
                severity="EMERGENCY",
                measured_value=voltage,
                reason=f"Overvoltage detected: {voltage:.1f}V exceeds threshold {FAULT_OVERVOLTAGE_V}V",
                evidence={
                    "voltage": voltage,
                    "threshold": FAULT_OVERVOLTAGE_V,
                    "pct_over": round((voltage - FAULT_OVERVOLTAGE_V) / FAULT_OVERVOLTAGE_V * 100, 1),
                },
                confidence=0.95,
            )
        )
        emergency_events.append(events[-1])

    # 2. Undervoltage
    voltage = latest_readings.get("voltage")
    if voltage is not None and voltage < FAULT_UNDERVOLTAGE_V:
        events.append(
            FaultEvent(
                pzem_number=pzem_number,
                timestamp=int(latest["timestamp"]),
                fault_type="undervoltage",
                severity="EMERGENCY",
                measured_value=voltage,
                reason=f"Undervoltage detected: {voltage:.1f}V below threshold {FAULT_UNDERVOLTAGE_V}V",
                evidence={
                    "voltage": voltage,
                    "threshold": FAULT_UNDERVOLTAGE_V,
                    "pct_under": round((FAULT_UNDERVOLTAGE_V - voltage) / FAULT_UNDERVOLTAGE_V * 100, 1),
                },
                confidence=0.95,
            )
        )
        emergency_events.append(events[-1])

    # 3. Overcurrent
    current = latest_readings.get("current")
    if current is not None and current > FAULT_OVERCURRENT_A:
        events.append(
            FaultEvent(
                pzem_number=pzem_number,
                timestamp=int(latest["timestamp"]),
                fault_type="overcurrent",
                severity="EMERGENCY",
                measured_value=current,
                reason=f"Overcurrent detected: {current:.2f}A exceeds threshold {FAULT_OVERCURRENT_A}A",
                evidence={
                    "current": current,
                    "threshold": FAULT_OVERCURRENT_A,
                    "pct_over": round((current - FAULT_OVERCURRENT_A) / FAULT_OVERCURRENT_A * 100, 1),
                },
                confidence=0.95,
            )
        )
        emergency_events.append(events[-1])

    # 4. Power-factor drop
    pf = latest_readings.get("pf")
    if pf is not None and pf < FAULT_PF_DROP:
        events.append(
            FaultEvent(
                pzem_number=pzem_number,
                timestamp=int(latest["timestamp"]),
                fault_type="power_factor_drop",
                severity="WARNING",
                measured_value=pf,
                reason=f"Power-factor drop detected: PF={pf:.3f} below threshold {FAULT_PF_DROP}",
                evidence={
                    "pf": pf,
                    "threshold": FAULT_PF_DROP,
                    "pct_below": round((FAULT_PF_DROP - pf) / FAULT_PF_DROP * 100, 1),
                },
                confidence=0.9,
            )
        )
        # PF drop is WARNING, not EMERGENCY, unless it's extreme
        # (we let the caller decide; here we keep it WARNING)

    # 5. Frequency deviation
    frequency = latest_readings.get("frequency")
    if frequency is not None:
        freq_deviation = abs(frequency - 50.0)  # assume 50 Hz grid
        if freq_deviation > FAULT_FREQ_DEVIATION_HZ:
            events.append(
                FaultEvent(
                    pzem_number=pzem_number,
                    timestamp=int(latest["timestamp"]),
                    fault_type="frequency_deviation",
                    severity="EMERGENCY" if freq_deviation > FAULT_FREQ_DEVIATION_HZ + 1 else "WARNING",
                    measured_value=frequency,
                    reason=f"Frequency deviation detected: {frequency:.2f}Hz "
                           f"deviation of {freq_deviation:.2f}Hz exceeds "
                           f"threshold {FAULT_FREQ_DEVIATION_HZ}Hz",
                    evidence={
                        "frequency": frequency,
                        "nominal_frequency": 50.0,
                        "deviation_hz": round(freq_deviation, 2),
                        "threshold_hz": FAULT_FREQ_DEVIATION_HZ,
                    },
                    confidence=0.9,
                )
            )
            emergency_events.append(events[-1])

    # 6. Abnormally high power
    power = latest_readings.get("power")
    if power is not None and power > FAULT_HIGH_POWER_W:
        events.append(
            FaultEvent(
                pzem_number=pzem_number,
                timestamp=int(latest["timestamp"]),
                fault_type="high_power",
                severity="EMERGENCY",
                measured_value=power,
                reason=f"Abnormally high power detected: {power:.1f}W exceeds threshold {FAULT_HIGH_POWER_W}W",
                evidence={
                    "power": power,
                    "threshold": FAULT_HIGH_POWER_W,
                    "pct_over": round((power - FAULT_HIGH_POWER_W) / FAULT_HIGH_POWER_W * 100, 1),
                },
                confidence=0.9,
            )
        )
        emergency_events.append(events[-1])

    # 7. Communication / data-quality failure
    # If the meter has very few valid rows even though status == READY,
    # that's a degradation warning. If status != READY, it's already
    # handled above.
    valid_rows = preprocess_result.valid_rows
    if valid_rows > 0 and valid_rows < 6:
        events.append(
            FaultEvent(
                pzem_number=pzem_number,
                timestamp=int(preprocess_result.newest_timestamp or 0),
                fault_type="communication_degraded",
                severity="WARNING",
                measured_value=float(valid_rows),
                reason=f"Communication/data-quality degradation: only {valid_rows} valid "
                       f"reading(s) available for PZEM {pzem_number}",
                evidence={"valid_rows": valid_rows},
                confidence=0.85,
            )
        )
        # WARNING, not EMERGENCY — meter is still reporting, just sparsely

    return emergency_events, events  # emergency_events first, then all events


# ---------------------------------------------------------------------------
# Pipeline: run fault diagnosis on all PZEMs
# ---------------------------------------------------------------------------

def run_fault_diagnosis_pipeline(
    preprocess_results: dict[int, PreprocessResult],
    settings: Optional[object] = None,
) -> dict[int, list[FaultEvent]]:
    """Run fault diagnosis on every PZEM and return per-PZEM event lists.

    Returns
    -------
    dict mapping PZEM number -> list of FaultEvent (all severities).
    The list is headed by EMERGENCY events first, then WARNING events.
    """
    settings = settings or get_settings()
    results: dict[int, list[FaultEvent]] = {}

    for pzem_number in range(1, settings.pzem_count + 1):
        pr = preprocess_results.get(pzem_number)
        if pr is None:
            # Should not happen if the preprocessing pipeline ran, but guard anyway
            results[pzem_number] = []
            continue

        emergency_events, all_events = diagnose_meter_faults(
            pzem_number, pr, settings=settings
        )
        # Return emergency events first, then all events
        results[pzem_number] = emergency_events + all_events

    return results


# ---------------------------------------------------------------------------
# Edge-case: fault that was active, then clears
# ---------------------------------------------------------------------------

def is_fault_active(events: list[FaultEvent], fault_type: str, pzem_number: int) -> bool:
    """Check if a specific fault type is currently active for a PZEM.

    A fault is "active" if the most recent event of that type has
    severity EMERGENCY or WARNING (not NORMAL / cleared).
    """
    # Find the latest event of this fault type for this PZEM
    latest = None
    for event in events:
        if event.pzem_number == pzem_number and event.fault_type == fault_type:
            if latest is None or event.timestamp > latest.timestamp:
                latest = event
    return latest is not None and latest.severity != "NORMAL"


def fault_cleared(events: list[FaultEvent], fault_type: str, pzem_number: int) -> bool:
    """Check if a specific fault has cleared (last event was NORMAL or no events)."""
    latest = None
    for event in events:
        if event.pzem_number == pzem_number and event.fault_type == fault_type:
            if latest is None or event.timestamp > latest.timestamp:
                latest = event
    return latest is None or latest.severity == "NORMAL"