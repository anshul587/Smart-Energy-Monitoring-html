"""
ai/diagnostic_recommendation.py
--------------------------------
Stage 4: Diagnostic & Recommendation Intelligence (Phase 2A).

Reads verified outputs from existing Stages 1-3 (fault_diagnosis) and
produces structured, evidence-based recommendations. Never invents
measurements, savings, timestamps, faults, or confidence.

Deterministic: identical fault result always produces identical
recommendation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .anomaly_detection import AnomalyDetectionResult
from .fault_diagnosis import FaultEvent
from .forecast import ForecastResult
from .maintenance_risk import RiskResult, SystemMaintenanceSummary
from .peak_detection import PeakResult, SystemPeakResult
from .energy_saving import Recommendation as EnergySavingRecommendation

# ---------------------------------------------------------------------------
# Supported fault types
# ---------------------------------------------------------------------------

SUPPORTED_FAULT_TYPES = (
    "overvoltage",
    "undervoltage",
    "overcurrent",
    "power_factor_drop",
    "frequency_deviation",
    "high_power",
    "communication_degraded",
)

# ---------------------------------------------------------------------------
# Deterministic fault-type-specific recommendation mappings
# ---------------------------------------------------------------------------

_FAULT_RULES: dict[str, dict[str, str]] = {
    "overvoltage": {
        "condition": "Voltage exceeds the nominal threshold.",
        "probable_cause": (
            "Possible incoming supply overvoltage, regulator malfunction, "
            "or measurement/calibration issue."
        ),
        "why_it_happened": (
            "Measured voltage exceeds the configured overvoltage threshold. "
            "This may indicate a supply-side regulation problem or a sensor "
            "calibration drift."
        ),
        "evidence": "voltage reading above threshold; compare with other PZEMs if available.",
        "what_to_check": (
            "Verify voltage with an independent meter; compare readings across "
            "multiple PZEMs; check incoming supply and voltage regulator."
        ),
        "what_to_do_now": (
            "Do NOT assume equipment failure. Verify the reading independently. "
            "If confirmed abnormal, arrange qualified electrical inspection."
        ),
        "corrective_action": (
            "Independent voltage verification first. If persistent abnormal "
            "voltage is confirmed, qualified electrical inspection required. "
            "Never conclude transformer failure from voltage alone."
        ),
        "urgency": "HIGH",
        "maintenance_required": True,
        "maintenance_timing": "Schedule inspection promptly; verify supply conditions immediately.",
    },
    "undervoltage": {
        "condition": "Voltage falls below the nominal threshold.",
        "probable_cause": (
            "Possible supply sag, excessive connected load, wiring/connection "
            "issue, or measurement problem."
        ),
        "why_it_happened": (
            "Measured voltage is below the undervoltage threshold. This may "
            "reflect a supply sag, overloaded circuit, or a loose/worn "
            "connection that increases resistance."
        ),
        "evidence": "voltage reading below threshold; correlate with current/load if available.",
        "what_to_check": (
            "Verify with an independent meter; compare with other PZEMs; check "
            "whether voltage drop correlates with high current or heavy load; "
            "inspect connections and wiring."
        ),
        "what_to_do_now": (
            "Verify the voltage drop with an independent meter. Check for "
            "correlation with high current. If persistent, arrange electrical inspection."
        ),
        "corrective_action": (
            "Independent voltage verification. If voltage drop correlates with "
            "high current, investigate load and connections. Persistent issue "
            "requires electrical inspection."
        ),
        "urgency": "HIGH",
        "maintenance_required": True,
        "maintenance_timing": "Schedule inspection promptly; verify supply and load conditions.",
    },
    "overcurrent": {
        "condition": "Current exceeds the rated threshold.",
        "probable_cause": (
            "Possible overload, excessive connected load, motor starting/inrush "
            "current, or equipment/circuit issue."
        ),
        "why_it_happened": (
            "Measured current exceeds the overcurrent threshold. This may "
            "indicate a genuine overload, inrush current during motor start, "
            "or a fault in connected equipment or the circuit."
        ),
        "evidence": "current reading above threshold; check active loads and breaker/cable ratings.",
        "what_to_check": (
            "Check active loads; verify breaker rating and cable rating; inspect "
            "load distribution; determine whether the overload is sustained or transient."
        ),
        "what_to_do_now": (
            "If genuinely overloaded, reduce or redistribute load. Do NOT claim a "
            "specific appliance is faulty without supporting evidence."
        ),
        "corrective_action": (
            "Reduce or redistribute load if overloaded. Verify breaker and cable "
            "ratings. Do not attribute fault to a specific appliance without evidence."
        ),
        "urgency": "HIGH",
        "maintenance_required": True,
        "maintenance_timing": "Address load immediately if sustained; inspect circuit ratings.",
    },
    "power_factor_drop": {
        "condition": "Power factor falls below the nominal threshold.",
        "probable_cause": (
            "Possible inductive loading (motors, transformers) or reactive-power "
            "issues. PF correction equipment may not be functioning."
        ),
        "why_it_happened": (
            "Power factor has dropped below the threshold, often indicating "
            "inductive loading from motors or transformers, or that PF correction "
            "equipment is not operating correctly."
        ),
        "evidence": "pf reading below threshold; identify which loads are operating.",
        "what_to_check": (
            "Identify which loads are currently operating; check whether PF "
            "correction equipment is functioning; do NOT blindly recommend "
            "capacitor installation."
        ),
        "what_to_do_now": (
            "Assess the operating loads and PF correction equipment status. "
            "Any PF correction recommendation must be conditional on a proper "
            "electrical assessment."
        ),
        "corrective_action": (
            "Verify operating loads and PF correction equipment status. "
            "Any PF correction recommendation is conditional on proper electrical "
            "assessment. Do not blindly recommend capacitor installation."
        ),
        "urgency": "MEDIUM",
        "maintenance_required": True,
        "maintenance_timing": "Assess during next scheduled inspection; PF correction if warranted after assessment.",
    },
    "frequency_deviation": {
        "condition": "Grid frequency deviates beyond the acceptable tolerance.",
        "probable_cause": (
            "Possible supply/source frequency abnormality or measurement issue."
        ),
        "why_it_happened": (
            "Measured frequency deviates beyond the tolerance band. This may "
            "reflect a supply/source frequency issue or a sensor/measurement "
            "anomaly."
        ),
        "evidence": "frequency reading outside tolerance; verify with independent source.",
        "what_to_check": (
            "Verify frequency with an independent meter or source; inspect "
            "supply/source if deviation is persistent. Do NOT claim a generator "
            "or utility fault without evidence."
        ),
        "what_to_do_now": (
            "Verify frequency with an independent source. If persistent deviation "
            "is confirmed, inspect the supply/source. Do not attribute fault to "
            "generator or utility without evidence."
        ),
        "corrective_action": (
            "Independent frequency verification. Persistent deviation requires "
            "supply/source inspection. No utility/generator fault claim without evidence."
        ),
        "urgency": "HIGH",
        "maintenance_required": True,
        "maintenance_timing": "Inspect supply/source promptly if deviation persists.",
    },
    "high_power": {
        "condition": "Power consumption is abnormally high.",
        "probable_cause": (
            "Possible unusually high load, simultaneous loads, or abnormal "
            "equipment operation."
        ),
        "why_it_happened": (
            "Measured power exceeds the high-power threshold. This may reflect "
            "an unusually high load, multiple simultaneous loads, or abnormal "
            "equipment operation."
        ),
        "evidence": "power reading above threshold; use only supplied evidence to assess.",
        "what_to_check": (
            "Check active loads; identify whether the high power is expected "
            "given current operating conditions. Do NOT invent the appliance "
            "responsible."
        ),
        "what_to_do_now": (
            "Use only supplied evidence to assess. Check active loads and "
            "determine whether the high power reading is expected. Do not "
            "attribute it to a specific appliance without evidence."
        ),
        "corrective_action": (
            "Assess whether high power is expected based on active loads. "
            "Do not identify a specific faulty appliance without supporting evidence."
        ),
        "urgency": "MEDIUM",
        "maintenance_required": False,
        "maintenance_timing": "Monitor; investigate only if high power is unexpected or sustained.",
    },
    "communication_degraded": {
        "condition": "PZEM communication or data quality is degraded.",
        "probable_cause": (
            "Possible sensor, communication, connectivity, or data-quality "
            "problem — not an electrical load fault."
        ),
        "why_it_happened": (
            "The PZEM has insufficient valid readings or sporadic data. This "
            "indicates a communication, connectivity, or data-quality issue "
            "rather than an electrical load problem."
        ),
        "evidence": "low valid row count or sporadic data availability.",
        "what_to_check": (
            "Check PZEM wiring, ESP32 connection, power supply, serial "
            "communication, and network/Firebase path as appropriate."
        ),
        "what_to_do_now": (
            "Inspect PZEM wiring and connections. Verify ESP32 power and serial "
            "communication. Check network/Firebase path. This is NOT an "
            "electrical load fault."
        ),
        "corrective_action": (
            "Check PZEM wiring, ESP32 connection, power, serial communication, "
            "and network/Firebase path. Resolve connectivity issue. "
            "Do not treat this as an electrical load fault."
        ),
        "urgency": "MEDIUM",
        "maintenance_required": True,
        "maintenance_timing": "Inspect wiring and connectivity at next opportunity.",
    },
}


# ---------------------------------------------------------------------------
# Anomaly + peak recommendation mappings
# ---------------------------------------------------------------------------

_ANOMALY_SEVERITY_MAP = {
    "HIGH_PROVISIONAL": ("EMERGENCY", "P1 - Critical"),
    "MEDIUM_PROVISIONAL": ("WARNING", "P2 - Important"),
    "LOW_PROVISIONAL": ("WARNING", "P2 - Important"),
}

_ANOMALY_RULES = {
    "condition": "Statistical anomaly detected — behavior deviates from this meter's own ACTIVE operating history.",
    "probable_cause": (
        "Possible unusual equipment behavior, abnormal load pattern, "
        "or sensor issue. Cause must remain conditional until "
        "investigated; an Isolation Forest flag means 'statistically "
        "unusual compared to this meter's own ACTIVE-state history'."
    ),
    "why_it_happened": (
        "Anomaly score exceeds the provisional threshold for this "
        "meter's own ACTIVE-state history. The Isolation Forest "
        "identified this reading as statistically unusual compared "
        "to the meter's own ACTIVE-state baseline."
    ),
    "evidence": "anomaly score, provisional severity, operating state from Isolation Forest.",
    "what_to_check": (
        "Check which loads were operating at the anomaly timestamp; "
        "verify sensor/connection status; compare with other PZEMs "
        "and with recent history for this meter; review operating "
        "state (ACTIVE/INACTIVE) and anomaly severity label."
    ),
    "what_to_do_now": (
        "Investigate what loads were operating at the anomaly time. "
        "Do NOT claim a specific appliance or component has failed. "
        "An anomaly flag indicates statistical unusualness, not a "
        "diagnosis of cause."
    ),
    "corrective_action": (
        "Investigate operating conditions at the anomaly timestamp. "
        "Verify sensor/connection integrity. Cause remains conditional "
        "until evidence supports a specific diagnosis. Do not assume "
        "component failure from an anomaly flag alone."
    ),
    "urgency": "MEDIUM",
    "maintenance_required": True,
    "maintenance_timing": "Investigate at next opportunity; escalate if anomalies persist or correlate with fault events.",
}

_PEAK_RULES_HIGH = {
    "condition": "Unusually high instantaneous power demand detected for this PZEM.",
    "probable_cause": (
        "Possible unusually high load, simultaneous loads, or "
        "abnormal equipment operation during the peak window."
    ),
    "why_it_happened": (
        "Power exceeded the baseline during the analysis window. "
        "The peak represents an observed maximum, not a fault — it "
        "indicates unusually high instantaneous demand."
    ),
    "evidence": "observed peak power value and timestamp from existing peak detection results.",
    "what_to_check": (
        "Check which loads were operating during the peak timestamp; "
        "determine whether the high power was expected given "
        "operating conditions; compare with average and baseline power."
    ),
    "what_to_do_now": (
        "Use only supplied evidence to assess. Check active loads "
        "during the peak period and determine whether the high power "
        "was expected. Do not invent the appliance responsible."
    ),
    "corrective_action": (
        "Assess whether high power was expected based on operating "
        "loads. Recommend load shifting or reduction only when "
        "supported by evidence. Do not identify a specific faulty "
        "appliance without supporting evidence."
    ),
    "urgency": "MEDIUM",
    "maintenance_required": False,
    "maintenance_timing": "Monitor; investigate only if peak is unexpected or recurring.",
}

_PEAK_RULES_RECURRING = {
    "condition": "Repeated high-power peak behavior detected across multiple periods.",
    "probable_cause": (
        "Possible repeated simultaneous/high-power loads or a "
        "persistent high-demand pattern. Recurrence suggests a "
        "systematic load pattern rather than a one-time event."
    ),
    "why_it_happened": (
        "Multiple peak events detected in the analysis window, "
        "indicating repeated high-power demand. The recurrence "
        "pattern and peak timestamps provide evidence of a "
        "consistent high-demand behavior."
    ),
    "evidence": "multiple peak timestamps and power values from existing peak detection results.",
    "what_to_check": (
        "Investigate repeated simultaneous/high-power loads; "
        "review peak timestamps for recurring time windows; "
        "check whether loads correlate across peak events."
    ),
    "what_to_do_now": (
        "Provide actionable load-management guidance. Investigate "
        "whether peak events occur in recurring time windows. Do "
        "not invent the appliance responsible for the peaks."
    ),
    "corrective_action": (
        "Implement load-management guidance based on peak timing "
        "patterns. Investigate whether high-power loads recur in "
        "consistent time windows. Do not attribute peaks to a "
        "specific appliance without evidence."
    ),
    "urgency": "MEDIUM",
    "maintenance_required": True,
    "maintenance_timing": "Analyze peak timing patterns; plan load-shifting measures.",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticRecommendation:
    """A structured, evidence-based diagnostic recommendation.

    All fields are derived deterministically from the input FaultEvent.
    No values are invented.
    """

    recommendation_id: str
    timestamp: int
    pzem_system: str
    condition: str
    fault_type: str
    severity: str
    priority: str
    probable_cause: str
    why_it_happened: str
    evidence: str
    what_to_check: str
    what_to_do_now: str
    corrective_action: str
    urgency: str
    maintenance_required: bool
    maintenance_timing: str
    energy_impact_kwh: Optional[float]
    cost_impact: Optional[float]
    confidence: float
    source_stages: tuple[str, ...]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """Deterministic recommendation engine.

    Converts verified FaultEvent objects into structured engineer-style
    recommendations. Pure function: identical input always produces
    identical output.
    """

    def recommend(self, fault: FaultEvent) -> DiagnosticRecommendation:
        """Convert a FaultEvent into a DiagnosticRecommendation.

        Parameters
        ----------
        fault : FaultEvent
            A verified fault result from Stage 3 (fault_diagnosis).

        Returns
        -------
        DiagnosticRecommendation
            Deterministic recommendation derived from the fault event.
        """
        fault_type = fault.fault_type
        if fault_type not in SUPPORTED_FAULT_TYPES:
            raise ValueError(
                f"Unsupported fault type: {fault_type!r}. "
                f"Supported types: {', '.join(SUPPORTED_FAULT_TYPES)}"
            )

        rule = _FAULT_RULES[fault_type]

        # Derive severity -> priority mapping deterministically
        severity = fault.severity
        priority = self._severity_to_priority(severity)

        # Use confidence from the fault event; fallback to a safe default
        confidence = fault.confidence if fault.confidence is not None else 0.5

        # Build recommendation_id deterministically
        rec_id = self._build_recommendation_id(
            pzem_number=fault.pzem_number,
            timestamp=fault.timestamp,
            fault_type=fault_type,
            severity=severity,
        )

        # Determine if fault result supplies deterministic energy/cost impact
        energy_impact = self._extract_deterministic_impact(fault.evidence or {})
        cost_impact = None  # No cost impact derived in Phase 2A

        return DiagnosticRecommendation(
            recommendation_id=rec_id,
            timestamp=fault.timestamp,
            pzem_system=f"PZEM-{fault.pzem_number}",
            condition=rule["condition"],
            fault_type=fault_type,
            severity=severity,
            priority=priority,
            probable_cause=rule["probable_cause"],
            why_it_happened=rule["why_it_happened"],
            evidence=rule["evidence"],
            what_to_check=rule["what_to_check"],
            what_to_do_now=rule["what_to_do_now"],
            corrective_action=rule["corrective_action"],
            urgency=rule["urgency"],
            maintenance_required=rule["maintenance_required"],
            maintenance_timing=rule["maintenance_timing"],
            energy_impact_kwh=energy_impact,
            cost_impact=cost_impact,
            confidence=confidence,
            source_stages=("Stage 3: Fault Diagnosis",),
        )

    @staticmethod
    def _severity_to_priority(severity: str) -> str:
        """Map fault severity to recommendation priority deterministically."""
        mapping = {
            "EMERGENCY": "P1 - Critical",
            "WARNING": "P2 - Important",
            "NORMAL": "P3 - Informational",
        }
        return mapping.get(severity, "P3 - Informational")

    @staticmethod
    def _build_recommendation_id(pzem_number: int, timestamp: int,
                                  fault_type: str, severity: str) -> str:
        """Build a deterministic, reproducible recommendation ID."""
        payload = json.dumps(
            {
                "pzem": pzem_number,
                "ts": timestamp,
                "fault": fault_type,
                "sev": severity,
            },
            sort_keys=True,
        )
        return f"REC-{hashlib.sha256(payload.encode()).hexdigest()[:12].upper()}"

    @staticmethod
    def _extract_deterministic_impact(evidence: dict) -> Optional[float]:
        """Extract energy impact ONLY if the fault result contains a
        deterministic impact value. Returns None otherwise."""
        if "energy_impact_kwh" in evidence:
            val = evidence["energy_impact_kwh"]
            if isinstance(val, (int, float)):
                return float(val)
        return None

    @staticmethod
    def recommend_batch(faults: list[FaultEvent]) -> list[DiagnosticRecommendation]:
        """Convert multiple FaultEvent objects into recommendations."""
        return [engine.recommend(f) for f in faults]

    # -------------------------------------------------------------------
    # Phase 2B: Anomaly + Peak reasoning
    # -------------------------------------------------------------------

    def recommend_from_anomaly(
        self, anomaly_result: AnomalyDetectionResult
    ) -> DiagnosticRecommendation:
        """Convert an AnomalyDetectionResult into a DiagnosticRecommendation.

        Uses the latest ANOMALY row from the result_frame.
        If no anomaly is found or the result is INSUFFICIENT_DATA,
        returns a recommendation reflecting that state.
        """
        pzem_number = anomaly_result.pzem_number
        timestamp = 0
        severity = "NORMAL"
        priority = "P3 - Informational"
        confidence = 0.0
        anomaly_score: Optional[float] = None
        anomaly_severity_label: Optional[str] = None
        condition = "No scored data available for anomaly assessment."
        probable_cause = "No result_frame available for anomaly scoring."
        why_it_happened = "No anomaly result frame was produced."
        evidence = "result_frame is None or empty."
        operating_state = "UNKNOWN"
        operating_state_method = anomaly_result.operating_state_method

        if anomaly_result.model_status == "INSUFFICIENT_DATA":
            condition = "Insufficient data for anomaly detection."
            probable_cause = (
                "Not enough active historical data to train an IsolationForest "
                "model for this meter. Cause cannot be determined."
            )
            why_it_happened = anomaly_result.reason or "Insufficient data for anomaly detection."
            evidence = (
                f"training_rows={anomaly_result.training_rows}, "
                f"active_days_represented={anomaly_result.active_days_represented}"
            )
        elif anomaly_result.result_frame is not None and not anomaly_result.result_frame.empty:
            scored = anomaly_result.result_frame[
                anomaly_result.result_frame["anomaly_label"] == "ANOMALY"
            ]
            if not scored.empty:
                latest = scored.iloc[-1]
                timestamp = int(latest["timestamp"])
                anomaly_score = float(latest["anomaly_score_normalized"])
                anomaly_severity_label = str(latest["anomaly_severity_provisional"])
                severity, priority = _ANOMALY_SEVERITY_MAP.get(
                    anomaly_severity_label, ("WARNING", "P2 - Important")
                )
                confidence = float(anomaly_score) if anomaly_score is not None else 0.5
                operating_state = str(latest.get("operating_state", "UNKNOWN"))
                condition = "Anomaly detected - statistically unusual behavior."
                why_it_happened = (
                    f"Anomaly detected with normalized score {anomaly_score:.4f} "
                    f"and provisional severity {anomaly_severity_label}. "
                    f"Operating state at time of anomaly: {operating_state}."
                )
                evidence = (
                    f"anomaly_score_normalized={anomaly_score:.4f}, "
                    f"severity={anomaly_severity_label}, "
                    f"operating_state={operating_state}"
                )
            else:
                condition = "No anomaly detected - latest scored row is NORMAL."
                probable_cause = (
                    "No statistically significant anomaly detected for "
                    "this meter's ACTIVE-state history."
                )
                why_it_happened = "Latest scored row was classified as NORMAL by the IsolationForest model."
                evidence = "No ANOMALY rows found in the result_frame."
                timestamp = int(anomaly_result.result_frame["timestamp"].iloc[-1]) if len(anomaly_result.result_frame) > 0 else 0

        rule = _ANOMALY_RULES

        rec_id = self._build_recommendation_id(
            pzem_number=pzem_number,
            timestamp=timestamp,
            fault_type="anomaly",
            severity=severity,
        )

        return DiagnosticRecommendation(
            recommendation_id=rec_id,
            timestamp=timestamp,
            pzem_system=f"PZEM-{pzem_number}",
            condition=condition,
            fault_type="anomaly",
            severity=severity,
            priority=priority,
            probable_cause=rule["probable_cause"],
            why_it_happened=why_it_happened,
            evidence=rule["evidence"] + f" {evidence}",
            what_to_check=rule["what_to_check"],
            what_to_do_now=rule["what_to_do_now"],
            corrective_action=rule["corrective_action"],
            urgency=rule["urgency"],
            maintenance_required=rule["maintenance_required"],
            maintenance_timing=rule["maintenance_timing"],
            energy_impact_kwh=None,
            cost_impact=None,
            confidence=confidence,
            source_stages=("Stage 3: Anomaly Detection",),
        )

    def recommend_from_peak(
        self, peak_result: PeakResult, peak_type: str = "high_peak"
    ) -> DiagnosticRecommendation:
        """Convert a PeakResult into a DiagnosticRecommendation.

        peak_type controls the recommendation framing:
        - 'high_peak': single high-power peak
        - 'recurring_peak': handled separately via recommend_from_recurring_peak
        """
        if peak_result.status != "PEAK_FOUND":
            condition = "No peak found in the analysis window."
            probable_cause = "No sustained peak detected above the isolation threshold."
            why_it_happened = peak_result.reason or "No PEAK_FOUND result."
            evidence = "status=NO_PEAK"
            timestamp = 0
        else:
            condition = _PEAK_RULES_HIGH["condition"]
            probable_cause = _PEAK_RULES_HIGH["probable_cause"]
            why_it_happened = (
                f"Peak power of {peak_result.peak_power_w:.2f} W detected at "
                f"timestamp {peak_result.peak_timestamp}. Baseline (median) "
                f"power was {peak_result.baseline_power_w:.2f} W "
                f"(+{peak_result.peak_above_baseline_w:.2f} W above). "
                f"Sustained: {peak_result.sustained}."
            )
            evidence = (
                f"peak_power_w={peak_result.peak_power_w}, "
                f"peak_timestamp={peak_result.peak_timestamp}, "
                f"sustained={peak_result.sustained}, "
                f"baseline_power_w={peak_result.baseline_power_w:.2f}"
            )
            timestamp = peak_result.peak_timestamp or 0

        rule = _PEAK_RULES_HIGH
        rec_id = self._build_recommendation_id(
            pzem_number=peak_result.pzem_number,
            timestamp=timestamp,
            fault_type=peak_type,
            severity="WARNING",
        )

        return DiagnosticRecommendation(
            recommendation_id=rec_id,
            timestamp=timestamp,
            pzem_system=f"PZEM-{peak_result.pzem_number}",
            condition=condition,
            fault_type=peak_type,
            severity="WARNING",
            priority="P2 - Important",
            probable_cause=probable_cause,
            why_it_happened=why_it_happened,
            evidence=rule["evidence"] + f" {evidence}",
            what_to_check=rule["what_to_check"],
            what_to_do_now=rule["what_to_do_now"],
            corrective_action=rule["corrective_action"],
            urgency=rule["urgency"],
            maintenance_required=rule["maintenance_required"],
            maintenance_timing=rule["maintenance_timing"],
            energy_impact_kwh=None,
            cost_impact=None,
            confidence=0.7 if peak_result.status == "PEAK_FOUND" else 0.3,
            source_stages=("Stage 7: Peak Detection",),
        )

    def recommend_from_system_peak(
        self, system_peak: SystemPeakResult
    ) -> DiagnosticRecommendation:
        """Convert a SystemPeakResult into a DiagnosticRecommendation."""
        if system_peak.status != "PEAK_FOUND":
            condition = "No system-wide peak found."
            probable_cause = "No simultaneous high-power peak detected across all analyzed PZEMs."
            why_it_happened = system_peak.reason or "No system-wide PEAK_FOUND result."
            evidence = "status=NO_PEAK"
            timestamp = 0
        else:
            condition = _PEAK_RULES_HIGH["condition"]
            probable_cause = (
                f"System-wide simultaneous peak of {system_peak.total_peak_power_w:.2f} W "
                f"detected at slot {system_peak.timestamp}. "
                f"Dominant PZEM(s): {system_peak.dominant_pzems}."
            )
            why_it_happened = (
                f"System-wide peak of {system_peak.total_peak_power_w:.2f} W observed "
                f"in 300s slot starting at {system_peak.timestamp}. "
                f"{system_peak.meters_analyzed} PZEM(s) contributed to this simultaneous peak."
            )
            evidence = (
                f"total_peak_power_w={system_peak.total_peak_power_w:.2f}, "
                f"timestamp={system_peak.timestamp}, "
                f"dominant_pzems={system_peak.dominant_pzems}, "
                f"meters_analyzed={system_peak.meters_analyzed}"
            )
            timestamp = system_peak.timestamp or 0

        rule = _PEAK_RULES_HIGH
        rec_id = self._build_recommendation_id(
            pzem_number=0,
            timestamp=timestamp,
            fault_type="system_peak",
            severity="WARNING",
        )

        return DiagnosticRecommendation(
            recommendation_id=rec_id,
            timestamp=timestamp,
            pzem_system="SYSTEM",
            condition=condition,
            fault_type="system_peak",
            severity="WARNING",
            priority="P2 - Important",
            probable_cause=probable_cause,
            why_it_happened=why_it_happened,
            evidence=rule["evidence"] + f" {evidence}",
            what_to_check=rule["what_to_check"],
            what_to_do_now=rule["what_to_do_now"],
            corrective_action=rule["corrective_action"],
            urgency=rule["urgency"],
            maintenance_required=rule["maintenance_required"],
            maintenance_timing=rule["maintenance_timing"],
            energy_impact_kwh=None,
            cost_impact=None,
            confidence=0.7 if system_peak.status == "PEAK_FOUND" else 0.3,
            source_stages=("Stage 7: Peak Detection (system)",),
        )

    def recommend_from_recurring_peak(
        self, peak_results: dict[int, PeakResult], system_peak: Optional[SystemPeakResult] = None
    ) -> DiagnosticRecommendation:
        """Convert multiple PeakResults into a recurring-peak recommendation.

        A recurring peak is identified when multiple PZEMs have
        PEAK_FOUND results, or when the same PZEM has peaks at
        similar time windows.
        """
        peak_found = [r for r in peak_results.values() if r.status == "PEAK_FOUND"]

        if len(peak_found) < 2 and (system_peak is None or system_peak.status != "PEAK_FOUND"):
            condition = "No recurring peak pattern detected."
            probable_cause = "Insufficient peak data to identify a recurring pattern."
            why_it_happened = "Fewer than 2 peak events found across the fleet."
            evidence = f"peak_found_count={len(peak_found)}"
            timestamp = 0
        else:
            condition = _PEAK_RULES_RECURRING["condition"]
            probable_cause = _PEAK_RULES_RECURRING["probable_cause"]
            peak_timestamps = [
                r.peak_timestamp for r in peak_found if r.peak_timestamp
            ]
            dominant_pzems = (
                system_peak.dominant_pzems if system_peak and system_peak.status == "PEAK_FOUND" else []
            )
            why_it_happened = (
                f"Recurring peak behavior detected: {len(peak_found)} PZEM(s) "
                f"with PEAK_FOUND results. Peak timestamps: {peak_timestamps}. "
                f"Dominant PZEM(s): {dominant_pzems}."
            )
            evidence = (
                f"recurring_peak_count={len(peak_found)}, "
                f"peak_timestamps={peak_timestamps}, "
                f"dominant_pzems={dominant_pzems}, "
                f"system_peak={'PEAK_FOUND' if system_peak and system_peak.status == 'PEAK_FOUND' else 'NO_PEAK'}"
            )
            timestamp = max(peak_timestamps) if peak_timestamps else 0

        rule = _PEAK_RULES_RECURRING
        rec_id = self._build_recommendation_id(
            pzem_number=0,
            timestamp=timestamp,
            fault_type="recurring_peak",
            severity="WARNING",
        )

        return DiagnosticRecommendation(
            recommendation_id=rec_id,
            timestamp=timestamp,
            pzem_system="SYSTEM",
            condition=condition,
            fault_type="recurring_peak",
            severity="WARNING",
            priority="P2 - Important",
            probable_cause=probable_cause,
            why_it_happened=why_it_happened,
            evidence=rule["evidence"] + f" {evidence}",
            what_to_check=rule["what_to_check"],
            what_to_do_now=rule["what_to_do_now"],
            corrective_action=rule["corrective_action"],
            urgency=rule["urgency"],
            maintenance_required=rule["maintenance_required"],
            maintenance_timing=rule["maintenance_timing"],
            energy_impact_kwh=None,
            cost_impact=None,
            confidence=0.6 if len(peak_found) >= 2 else 0.4,
            source_stages=("Stage 7: Peak Detection",),
        )


# -------------------------------------------------------------------
    # Phase 2C: Cross-pipeline, energy impact, cost, bill,
    #           energy-saving integration, maintenance, forecast+peak
    # -------------------------------------------------------------------

    ENERGY_IMPACT_MEASURED = "MEASURED"
    ENERGY_IMPACT_ESTIMATED = "ESTIMATED"
    ENERGY_IMPACT_PROJECTED = "PROJECTED"
    ENERGY_IMPACT_UNKNOWN = "UNKNOWN"

    @staticmethod
    def _classify_energy_impact(
        energy_kwh: Optional[float], source: str
    ) -> Tuple[str, Optional[float]]:
        """Classify energy impact based on supplied data.

        Returns (classification, kwh_value).
        MEASURED: value supplied directly from source data.
        ESTIMATED: value derivable from existing measurements via math.
        PROJECTED: value from forecast/model.
        UNKNOWN: no valid data supplied.
        """
        if energy_kwh is not None and isinstance(energy_kwh, (int, float)) and energy_kwh >= 0:
            if source == "actual":
                return ("MEASURED", float(energy_kwh))
            elif source == "measured":
                return ("MEASURED", float(energy_kwh))
            elif source == "estimated":
                return ("ESTIMATED", float(energy_kwh))
            elif source == "forecast":
                return ("PROJECTED", float(energy_kwh))
            else:
                return ("UNKNOWN", float(energy_kwh))
        return ("UNKNOWN", None)

    @staticmethod
    def _calculate_cost(
        energy_kwh: Optional[float], tariff: Optional[float]
    ) -> Tuple[Optional[float], str]:
        """Calculate cost = energy_kwh × tariff when both valid.

        Returns (cost, status).
        status is "CALCULATED" or "UNKNOWN".
        """
        if (energy_kwh is not None and isinstance(energy_kwh, (int, float))
                and tariff is not None and isinstance(tariff, (int, float))
                and energy_kwh >= 0 and tariff > 0
                and math.isfinite(energy_kwh)
                and math.isfinite(tariff)):
            return (float(energy_kwh) * float(tariff), "CALCULATED")
        return (None, "UNKNOWN")

    def recommend_from_cross_pipeline(
        self,
        fault_result: Optional[FaultEvent] = None,
        peak_result: Optional[PeakResult] = None,
        anomaly_result: Optional[AnomalyDetectionResult] = None,
        energy_kwh: Optional[float] = None,
        tariff: Optional[float] = None,
        dominant_pzem: Optional[int] = None,
    ) -> DiagnosticRecommendation:
        """Combine multiple verified pipeline results into ONE richer
        diagnostic recommendation.

        When high current + high power + recurring peak overlap,
        produce a combined overloaded-load assessment.
        """
        signals = []
        if fault_result is not None:
            signals.append(f"fault: {fault_result.fault_type} ({fault_result.severity})")
        if peak_result is not None and peak_result.status == "PEAK_FOUND":
            signals.append(f"peak: {peak_result.peak_power_w:.0f}W")
        if anomaly_result is not None and anomaly_result.model_status == "READY":
            scored = anomaly_result.result_frame[
                anomaly_result.result_frame["anomaly_label"] == "ANOMALY"
            ] if anomaly_result.result_frame is not None else None
            if scored is not None and not scored.empty:
                signals.append(f"anomaly: {len(scored)} ANOMALY rows")

        signal_text = "; ".join(signals) if signals else "No pipeline signals supplied"

        # Determine probable condition
        if len(signals) >= 2:
            condition = "Combined signals suggest possible overloaded or stressed circuit."
            probable_cause = (
                "Multiple pipeline results (fault + peak and/or anomaly) "
                "converge on a possible excessive or overloaded load condition. "
                "Cause remains conditional until investigated."
            )
        elif len(signals) == 1:
            condition = signals[0].split(":")[0].capitalize() + " condition detected."
            probable_cause = _FAULT_RULES.get(
                fault_result.fault_type if fault_result else "high_power", {}
            ).get("probable_cause", "Single signal requires investigation.")
        else:
            condition = "No cross-pipeline signals to assess."
            probable_cause = "Insufficient evidence from pipeline results."

        # Determine urgency from fault severity
        urgency = "MEDIUM"
        if fault_result is not None:
            urgency = "HIGH" if fault_result.severity == "EMERGENCY" else "MEDIUM"
        elif peak_result is not None and peak_result.status == "PEAK_FOUND":
            urgency = "MEDIUM"

        # Energy impact
        impact_class, impact_kwh = self._classify_energy_impact(
            energy_kwh, source="measured" if energy_kwh is not None else "unknown"
        )
        cost, cost_status = self._calculate_cost(impact_kwh, tariff)

        what_to_check = (
            "Check active loads, breaker/cable ratings, load distribution, "
            "and identify whether the high power is expected."
        )
        what_to_do_now = (
            "Reduce or redistribute load if confirmed overloaded. "
            "Do NOT claim a specific appliance is faulty without evidence."
        )

        rec_id = self._build_recommendation_id(
            pzem_number=dominant_pzem or 0,
            timestamp=peak_result.peak_timestamp if peak_result and peak_result.peak_timestamp else 0,
            fault_type="cross_pipeline",
            severity="WARNING",
        )

        return DiagnosticRecommendation(
            recommendation_id=rec_id,
            timestamp=peak_result.peak_timestamp if peak_result and peak_result.peak_timestamp else 0,
            pzem_system=f"PZEM-{dominant_pzem}" if dominant_pzem else "SYSTEM",
            condition=condition,
            fault_type="cross_pipeline",
            severity="WARNING",
            priority="P2 - Important",
            probable_cause=probable_cause,
            why_it_happened=f"Cross-pipeline convergence: {signal_text}.",
            evidence=f"Signals: {signal_text}. Energy impact: {impact_class}.",
            what_to_check=what_to_check,
            what_to_do_now=what_to_do_now,
            corrective_action=(
                "Address all converging signals together. Reduce/redistribute load "
                "if confirmed. Do not attribute to a specific appliance without evidence."
            ),
            urgency=urgency,
            maintenance_required=True,
            maintenance_timing="Investigate converging signals promptly.",
            energy_impact_kwh=impact_kwh,
            cost_impact=cost,
            confidence=0.7 if len(signals) >= 2 else 0.5,
            source_stages=tuple(stage for stage in [
            "Stage 3: Fault Diagnosis" if fault_result is not None else None,
            "Stage 3: Anomaly Detection" if anomaly_result is not None and anomaly_result.model_status == "READY" and (anomaly_result.result_frame is not None and not anomaly_result.result_frame.empty) else None,
            "Stage 7: Peak Detection" if peak_result is not None and peak_result.status == "PEAK_FOUND" else None,
        ] if stage),
        )

    def recommend_from_bill(
        self,
        bill_result: Optional[dict] = None,
        peak_result: Optional[PeakResult] = None,
        dominant_pzem: Optional[int] = None,
        anchor_timestamp: Optional[int] = None,
    ) -> DiagnosticRecommendation:
        """Generate a bill recommendation from existing bill prediction
        and peak results.

        anchor_timestamp: explicit timestamp for the recommendation.
        Falls back to bill_result["anchor_timestamp"] if present,
        otherwise None (meaning no valid timestamp available).
        """
        if bill_result is None or bill_result.get("status") != "OK":
            condition = "Bill data unavailable or insufficient."
            probable_cause = "Insufficient data for bill prediction."
            why_it_happened = bill_result.get("reason", "No valid bill prediction available.") if bill_result else "No bill result supplied."
            evidence = "bill_result is None or status != OK"
            rec_id = self._build_recommendation_id(
                pzem_number=dominant_pzem or 0,
                timestamp=anchor_timestamp or 0,
                fault_type="bill",
                severity="NORMAL",
            )
            return DiagnosticRecommendation(
                recommendation_id=rec_id,
                timestamp=anchor_timestamp or 0,
                pzem_system=f"PZEM-{dominant_pzem}" if dominant_pzem else "SYSTEM",
                condition=condition, fault_type="bill",
                severity="NORMAL", priority="P3 - Informational",
                probable_cause=probable_cause,
                why_it_happened=why_it_happened,
                evidence=evidence,
                what_to_check="Verify bill data availability.",
                what_to_do_now="Obtain valid bill prediction data.",
                corrective_action="Re-run bill prediction with sufficient data.",
                urgency="MEDIUM", maintenance_required=False,
                maintenance_timing="N/A",
                energy_impact_kwh=None, cost_impact=None,
                confidence=0.3,
                source_stages=("Stage 10: Bill Prediction",),
            )

        actual_kwh = bill_result.get("actual_energy_kwh")
        forecast_kwh = bill_result.get("forecast_energy_kwh")
        estimated_total = bill_result.get("estimated_total_energy_kwh")
        rate = bill_result.get("rate")
        estimated_bill = bill_result.get("estimated_bill")

        impact_class, impact_kwh = self._classify_energy_impact(
            estimated_total, source="estimated"
        )
        cost, cost_status = self._calculate_cost(estimated_total, rate)

        peak_text = ""
        if peak_result is not None and peak_result.status == "PEAK_FOUND":
            peak_text = f"Peak power {peak_result.peak_power_w:.0f}W observed."

        condition = "Bill impact assessment based on actual and forecasted consumption."
        probable_cause = (
            f"Actual consumption: {actual_kwh:.3f} kWh, forecast: "
            f"{forecast_kwh:.3f} kWh, estimated total: {estimated_total:.3f} kWh. "
            f"{peak_text}"
        )
        rate_text = f"rate={rate}/kWh, " if rate is not None else ""
        why_it_happened = (
            f"Bill prediction: actual={actual_kwh:.3f}kWh, "
            f"forecast={forecast_kwh:.3f}kWh, total={estimated_total:.3f}kWh, "
            f"{rate_text}estimated_bill={estimated_bill}."
        )

        rec_id = self._build_recommendation_id(
            pzem_number=dominant_pzem or 0,
            timestamp=anchor_timestamp or bill_result.get("anchor_timestamp") or 0,
            fault_type="bill",
            severity="WARNING",
        )

        return DiagnosticRecommendation(
            recommendation_id=rec_id,
            timestamp=anchor_timestamp or bill_result.get("anchor_timestamp") or 0,
            pzem_system=f"PZEM-{dominant_pzem}" if dominant_pzem else "SYSTEM",
            condition=condition,
            fault_type="bill",
            severity="WARNING",
            priority="P2 - Important",
            probable_cause=probable_cause,
            why_it_happened=why_it_happened,
            evidence=f"actual={actual_kwh}, forecast={forecast_kwh}, total={estimated_total}, {rate_text}bill={estimated_bill}. Impact: {impact_class}.",
            what_to_check=(
                "Check actual consumption patterns, identify high-consumption "
                "time windows, verify meter readings. Do not invent which "
                "appliance caused the bill."
            ),
            what_to_do_now=(
                "Review consumption evidence. Consider load shifting/reduction "
                "for high-consumption periods. Do not attribute bill to a "
                "specific appliance without evidence."
            ),
            corrective_action=(
                "Review consumption evidence and identify actionable "
                "load-management opportunities. Do not assign fault to a "
                "specific appliance without supporting evidence."
            ),
            urgency="MEDIUM",
            maintenance_required=False,
            maintenance_timing="N/A",
            energy_impact_kwh=impact_kwh,
            cost_impact=cost,
            confidence=0.7 if estimated_total is not None else 0.3,
            source_stages=("Stage 10: Bill Prediction",),
        )

    def recommend_from_energy_saving(
        self,
        es_recommendation: Optional[EnergySavingRecommendation] = None,
    ) -> DiagnosticRecommendation:
        """Convert an existing energy-saving Recommendation into the
        Stage 4 DiagnosticRecommendation structure."""
        if es_recommendation is None:
            condition = "No energy-saving recommendation available."
            probable_cause = "No existing energy-saving analysis to convert."
            why_it_happened = "No energy-saving recommendation supplied."
            evidence = "es_recommendation is None"
            rec_id = self._build_recommendation_id(
                pzem_number=0, timestamp=0, fault_type="energy_saving", severity="NORMAL"
            )
            return DiagnosticRecommendation(
                recommendation_id=rec_id, timestamp=0,
                pzem_system="SYSTEM", condition=condition,
                fault_type="energy_saving", severity="NORMAL",
                priority="P3 - Informational",
                probable_cause=probable_cause,
                why_it_happened=why_it_happened,
                evidence=evidence,
                what_to_check="Verify energy-saving analysis.",
                what_to_do_now="Obtain valid energy-saving recommendation.",
                corrective_action="Re-run energy-saving analysis.",
                urgency="MEDIUM", maintenance_required=False,
                maintenance_timing="N/A",
                energy_impact_kwh=None, cost_impact=None,
                confidence=0.3,
                source_stages=("Stage 11: Energy Saving",),
            )

        potential_saving = es_recommendation.potential_saving_kwh
        potential_cost = es_recommendation.potential_cost_saving
        priority = es_recommendation.priority
        rec_type = es_recommendation.recommendation_type

        impact_class, impact_kwh = self._classify_energy_impact(
            potential_saving, source="estimated" if potential_saving is not None else "unknown"
        )
        cost, cost_status = self._calculate_cost(potential_saving, None)

        _priority_map = {"HIGH": "P1 - Critical", "MEDIUM": "P2 - Important", "LOW": "P3 - Informational"}
        normalized_priority = _priority_map.get(priority, "P3 - Informational")

        condition = f"Energy-saving recommendation: {rec_type}."
        probable_cause = es_recommendation.reason
        why_it_happened = f"Existing energy-saving analysis: {rec_type} — {es_recommendation.reason}"
        evidence = f"type={rec_type}, priority={priority}, metrics={es_recommendation.supporting_metrics}"

        rec_id = self._build_recommendation_id(
            pzem_number=es_recommendation.pzem_number or 0,
            timestamp=es_recommendation.timestamp,
            fault_type="energy_saving",
            severity="WARNING" if priority == "HIGH" else "NORMAL",
        )

        return DiagnosticRecommendation(
            recommendation_id=rec_id,
            timestamp=es_recommendation.timestamp,
            pzem_system=f"PZEM-{es_recommendation.pzem_number}" if es_recommendation.pzem_number is not None else "SYSTEM",
            condition=condition,
            fault_type="energy_saving",
            severity="WARNING" if priority == "HIGH" else "NORMAL",
            priority=normalized_priority,
            probable_cause=probable_cause,
            why_it_happened=why_it_happened,
            evidence=evidence,
            what_to_check=(
                "Review the energy-saving recommendation's supporting metrics "
                "and evidence window. Do not invent which appliance is responsible."
            ),
            what_to_do_now=es_recommendation.recommendation,
            corrective_action=(
                f"Implement: {es_recommendation.recommendation}. "
                f"Potential saving: {potential_saving} kWh" if potential_saving is not None
                else f"Implement: {es_recommendation.recommendation}."
            ),
            urgency="MEDIUM" if priority == "MEDIUM" else "HIGH" if priority == "HIGH" else "LOW",
            maintenance_required=False,
            maintenance_timing="N/A",
            energy_impact_kwh=impact_kwh,
            cost_impact=potential_cost if potential_cost is not None else None,
            confidence=0.6,
            source_stages=("Stage 11: Energy Saving",),
        )

    def recommend_from_maintenance(
        self,
        risk_result: Optional[RiskResult] = None,
        system_summary: Optional[SystemMaintenanceSummary] = None,
    ) -> DiagnosticRecommendation:
        """Create a maintenance recommendation from existing maintenance-risk results."""
        if risk_result is None or risk_result.status != "RISK_ASSESSED":
            condition = "No maintenance risk assessment available."
            probable_cause = "Insufficient data for maintenance risk analysis."
            why_it_happened = risk_result.reason if risk_result and risk_result.reason else "No maintenance risk result supplied."
            evidence = "risk_result is None or status != RISK_ASSESSED"
            rec_id = self._build_recommendation_id(
                pzem_number=risk_result.pzem_number if risk_result else 0,
                timestamp=risk_result.window_end_ts if risk_result and risk_result.window_end_ts else 0,
                fault_type="maintenance",
                severity="NORMAL",
            )
            return DiagnosticRecommendation(
                recommendation_id=rec_id,
                timestamp=risk_result.window_end_ts if risk_result and risk_result.window_end_ts else 0,
                pzem_system=f"PZEM-{risk_result.pzem_number}" if risk_result and risk_result.pzem_number else "SYSTEM",
                condition=condition, fault_type="maintenance",
                severity="NORMAL", priority="P3 - Informational",
                probable_cause=probable_cause,
                why_it_happened=why_it_happened,
                evidence=evidence,
                what_to_check="Verify maintenance risk data availability.",
                what_to_do_now="Obtain valid maintenance risk assessment.",
                corrective_action="Re-run maintenance risk analysis.",
                urgency="MEDIUM", maintenance_required=False,
                maintenance_timing="N/A",
                energy_impact_kwh=None, cost_impact=None,
                confidence=0.3,
                source_stages=("Stage 8: Maintenance Risk",),
            )

        risk_score = risk_result.risk_score
        risk_level = risk_result.risk_level
        indicators = risk_result.indicators or []

        # Map risk level to urgency and maintenance timing
        urgency = "MEDIUM"
        maintenance_timing = "Schedule inspection at next opportunity."
        maintenance_required = True

        if risk_level == "CRITICAL":
            urgency = "HIGH"
            maintenance_timing = "Urgent inspection recommended."
        elif risk_level == "HIGH":
            urgency = "HIGH"
            maintenance_timing = "Maintenance recommended promptly."
        elif risk_level == "WARNING":
            urgency = "MEDIUM"
            maintenance_timing = "Maintenance recommended at next inspection."
        elif risk_level == "WATCH":
            urgency = "LOW"
            maintenance_timing = "Monitor at next inspection."
        else:
            urgency = "LOW"
            maintenance_required = False
            maintenance_timing = "Monitor only."

        # Determine energy impact from risk evidence
        impact_kwh = None
        for ev in (risk_result.evidence or []):
            if isinstance(ev, dict) and "energy" in str(ev):
                impact_kwh = None  # Risk evidence doesn't supply direct kWh
                break

        condition = f"Maintenance risk: {risk_level} (score {risk_score}/100)."
        probable_cause = (
            f"Risk score {risk_score}/100 based on indicators: "
            f"{', '.join(indicators[:3]) if indicators else 'none triggered'}."
        )
        why_it_happened = (
            f"Maintenance risk assessment: score={risk_score}, "
            f"level={risk_level}, sufficiency={risk_result.data_sufficiency}. "
            f"Indicators: {'; '.join(indicators[:5]) if indicators else 'none'}."
        )
        evidence = f"risk_score={risk_score}, level={risk_level}, indicators={len(indicators)}"

        rec_id = self._build_recommendation_id(
            pzem_number=risk_result.pzem_number,
            timestamp=risk_result.window_end_ts or 0,
            fault_type="maintenance",
            severity="WARNING" if risk_level in ("HIGH", "CRITICAL") else "NORMAL",
        )

        return DiagnosticRecommendation(
            recommendation_id=rec_id,
            timestamp=risk_result.window_end_ts or 0,
            pzem_system=f"PZEM-{risk_result.pzem_number}",
            condition=condition,
            fault_type="maintenance",
            severity="WARNING" if risk_level in ("HIGH", "CRITICAL") else "NORMAL",
            priority=f"P1 - Critical" if risk_level == "CRITICAL" else "P2 - Important" if risk_level in ("HIGH", "WARNING") else "P3 - Informational",
            probable_cause=probable_cause,
            why_it_happened=why_it_happened,
            evidence=evidence,
            what_to_check=(
                "Review risk indicators and evidence. Check the specific "
                "indicators that triggered the risk score. Do not invent "
                "a specific maintenance date."
            ),
            what_to_do_now=(
                f"Schedule {maintenance_timing.lower()} based on the risk "
                f"level. Investigate the triggered indicators."
            ),
            corrective_action=(
                f"Address triggered indicators: {'; '.join(indicators[:3]) if indicators else 'general maintenance check'}. "
                f"Do not invent an exact maintenance date."
            ),
            urgency=urgency,
            maintenance_required=maintenance_required,
            maintenance_timing=maintenance_timing,
            energy_impact_kwh=None,
            cost_impact=None,
            confidence=0.7 if risk_score and risk_score > 0 else 0.3,
            source_stages=("Stage 8: Maintenance Risk",),
        )

    def recommend_from_forecast_peak(
        self,
        forecast_result: Optional[ForecastResult] = None,
        peak_result: Optional[PeakResult] = None,
        system_peak: Optional[SystemPeakResult] = None,
    ) -> DiagnosticRecommendation:
        """Produce a combined planning recommendation when forecast
        indicates elevated future demand AND existing peak/high-power
        evidence supports it."""
        forecast_high = False
        forecast_text = ""
        if forecast_result is not None and forecast_result.status == "FORECAST":
            h24 = forecast_result.forecast_24h
            if isinstance(h24, dict) and h24.get("status") == "FORECAST":
                powers = h24.get("forecast_power_w", [])
                if powers and max(powers) > 0:
                    forecast_high = True
                    forecast_text = f"Forecast shows elevated demand (max {max(powers):.0f}W, confidence={h24.get('confidence')})."

        peak_high = peak_result is not None and peak_result.status == "PEAK_FOUND"
        sys_peak_high = system_peak is not None and system_peak.status == "PEAK_FOUND"

        if forecast_high and (peak_high or sys_peak_high):
            condition = "Forecast elevated demand + existing peak evidence → planning recommendation."
            probable_cause = (
                "Forecast indicates elevated future demand and existing peak "
                "evidence supports this pattern. Likely repeated simultaneous loads."
            )
            why_it_happened = (
                f"Forecast elevated: {forecast_text} "
                f"Peak evidence: {'observed' if peak_high else 'system peak'}. "
                f"Combined planning recommendation."
            )
            evidence = (
                f"forecast_high={forecast_high}, peak_high={peak_high}, "
                f"sys_peak_high={sys_peak_high}"
            )
            what_to_check = (
                "Investigate expected simultaneous loads during forecast windows. "
                "Identify flexible loads that could be scheduled or shifted. "
                "Do not invent the appliance responsible."
            )
            what_to_do_now = (
                "Consider scheduling/shifting flexible loads ahead of "
                "predicted high-demand windows. Prepare for peak demand."
            )
            corrective_action = (
                "Plan load-shifting for flexible loads during forecast peak windows. "
                "Investigate simultaneous load patterns. No appliance identity claimed."
            )
            urgency = "MEDIUM"
            maintenance_required = False
            maintenance_timing = "N/A"
            confidence = 0.7
        elif forecast_high:
            condition = "Forecast indicates elevated future demand."
            probable_cause = "Forecast shows elevated power demand in future windows."
            why_it_happened = f"Forecast elevated demand detected. {forecast_text}"
            evidence = f"forecast_high={forecast_high}, no peak evidence."
            what_to_check = "Investigate forecasted demand pattern and prepare accordingly."
            what_to_do_now = "Prepare for potential high-demand periods. Monitor actual consumption."
            corrective_action = "Plan for potential load management during forecast windows."
            urgency = "LOW"
            maintenance_required = False
            maintenance_timing = "N/A"
            confidence = 0.5
        else:
            condition = "No forecast+peak combined signal detected."
            probable_cause = "Insufficient evidence for combined planning recommendation."
            why_it_happened = "Either no forecast elevation or no peak evidence."
            evidence = "forecast_high=False or peak evidence absent."
            what_to_check = "Monitor forecast and peak data for convergence."
            what_to_do_now = "No combined action warranted at this time."
            corrective_action = "Continue monitoring forecast and peak indicators."
            urgency = "LOW"
            maintenance_required = False
            maintenance_timing = "N/A"
            confidence = 0.3

        rec_id = self._build_recommendation_id(
            pzem_number=peak_result.pzem_number if peak_result else 0,
            timestamp=peak_result.peak_timestamp if peak_result and peak_result.peak_timestamp else 0,
            fault_type="forecast_peak",
            severity="WARNING" if forecast_high and (peak_high or sys_peak_high) else "NORMAL",
        )

        return DiagnosticRecommendation(
            recommendation_id=rec_id,
            timestamp=peak_result.peak_timestamp if peak_result and peak_result.peak_timestamp else 0,
            pzem_system=f"PZEM-{peak_result.pzem_number}" if peak_result else "SYSTEM",
            condition=condition,
            fault_type="forecast_peak",
            severity="WARNING" if forecast_high and (peak_high or sys_peak_high) else "NORMAL",
            priority="P2 - Important" if forecast_high and (peak_high or sys_peak_high) else "P3 - Informational",
            probable_cause=probable_cause,
            why_it_happened=why_it_happened,
            evidence=evidence,
            what_to_check=what_to_check,
            what_to_do_now=what_to_do_now,
            corrective_action=corrective_action,
            urgency=urgency,
            maintenance_required=maintenance_required,
            maintenance_timing=maintenance_timing,
            energy_impact_kwh=None,
            cost_impact=None,
            confidence=confidence,
            source_stages=("Stage 9: Forecast", "Stage 7: Peak Detection"),
        )

    def deduplicate_recommendations(
        self, recommendations: List[DiagnosticRecommendation]
    ) -> List[DiagnosticRecommendation]:
        """Combine recommendations that describe the same underlying condition.

        Keeps distinct recommendations for distinct conditions.
        Merges recommendations with same (pzem_system, fault_type, severity).
        """
        seen_conditions = {}
        unique = []
        for rec in recommendations:
            dedup_key = (rec.pzem_system, rec.fault_type, rec.severity)
            if dedup_key in seen_conditions:
                existing = seen_conditions[dedup_key]
                idx = next(i for i, r in enumerate(unique) if r.recommendation_id == existing.recommendation_id)
                merged = DiagnosticRecommendation(
                    recommendation_id=existing.recommendation_id,
                    timestamp=existing.timestamp,
                    pzem_system=existing.pzem_system,
                    condition=existing.condition,
                    fault_type=existing.fault_type,
                    severity=existing.severity,
                    priority=existing.priority,
                    probable_cause=existing.probable_cause,
                    why_it_happened=f"{existing.why_it_happened} ALSO: {rec.why_it_happened}",
                    evidence=f"{existing.evidence} AND {rec.evidence}",
                    what_to_check=existing.what_to_check,
                    what_to_do_now=existing.what_to_do_now,
                    corrective_action=existing.corrective_action,
                    urgency=existing.urgency,
                    maintenance_required=existing.maintenance_required,
                    maintenance_timing=existing.maintenance_timing,
                    energy_impact_kwh=existing.energy_impact_kwh,
                    cost_impact=existing.cost_impact,
                    confidence=max(existing.confidence, rec.confidence),
                    source_stages=existing.source_stages + rec.source_stages,
                )
                unique[idx] = merged
                seen_conditions[dedup_key] = merged
                continue
            seen_conditions[dedup_key] = rec
            unique.append(rec)
        return unique


# Module-level engine instance for convenience
engine = RecommendationEngine()


def recommend(fault: FaultEvent) -> DiagnosticRecommendation:
    """Convenience function: convert a FaultEvent into a DiagnosticRecommendation."""
    return engine.recommend(fault)


def recommend_from_anomaly(anomaly_result: AnomalyDetectionResult) -> DiagnosticRecommendation:
    """Convenience: convert AnomalyDetectionResult to DiagnosticRecommendation."""
    return engine.recommend_from_anomaly(anomaly_result)


def recommend_from_peak(peak_result: PeakResult) -> DiagnosticRecommendation:
    """Convenience: convert PeakResult to DiagnosticRecommendation."""
    return engine.recommend_from_peak(peak_result)


def recommend_from_system_peak(system_peak: SystemPeakResult) -> DiagnosticRecommendation:
    """Convenience: convert SystemPeakResult to DiagnosticRecommendation."""
    return engine.recommend_from_system_peak(system_peak)


def recommend_from_recurring_peak(
    peak_results: dict[int, PeakResult],
    system_peak: Optional[SystemPeakResult] = None,
) -> DiagnosticRecommendation:
    """Convenience: convert PeakResults to recurring-peak DiagnosticRecommendation."""
    return engine.recommend_from_recurring_peak(peak_results, system_peak)


def recommend_from_cross_pipeline(
    fault_result: Optional[FaultEvent] = None,
    peak_result: Optional[PeakResult] = None,
    anomaly_result: Optional[AnomalyDetectionResult] = None,
    energy_kwh: Optional[float] = None,
    tariff: Optional[float] = None,
    dominant_pzem: Optional[int] = None,
) -> DiagnosticRecommendation:
    """Convenience: cross-pipeline recommendation."""
    return engine.recommend_from_cross_pipeline(
        fault_result=fault_result, peak_result=peak_result,
        anomaly_result=anomaly_result, energy_kwh=energy_kwh,
        tariff=tariff, dominant_pzem=dominant_pzem,
    )


def recommend_from_bill(
    bill_result: Optional[dict] = None,
    peak_result: Optional[PeakResult] = None,
    dominant_pzem: Optional[int] = None,
    anchor_timestamp: Optional[int] = None,
) -> DiagnosticRecommendation:
    """Convenience: bill recommendation."""
    return engine.recommend_from_bill(
        bill_result=bill_result, peak_result=peak_result,
        dominant_pzem=dominant_pzem,
        anchor_timestamp=anchor_timestamp,
    )


def recommend_from_energy_saving(
    es_recommendation: Optional[EnergySavingRecommendation] = None,
) -> DiagnosticRecommendation:
    """Convenience: energy-saving integration."""
    return engine.recommend_from_energy_saving(es_recommendation=es_recommendation)


def recommend_from_maintenance(
    risk_result: Optional[RiskResult] = None,
    system_summary: Optional[SystemMaintenanceSummary] = None,
) -> DiagnosticRecommendation:
    """Convenience: maintenance intelligence."""
    return engine.recommend_from_maintenance(
        risk_result=risk_result, system_summary=system_summary,
    )


def recommend_from_forecast_peak(
    forecast_result: Optional[ForecastResult] = None,
    peak_result: Optional[PeakResult] = None,
    system_peak: Optional[SystemPeakResult] = None,
) -> DiagnosticRecommendation:
    """Convenience: forecast + peak cross-reasoning."""
    return engine.recommend_from_forecast_peak(
        forecast_result=forecast_result, peak_result=peak_result,
        system_peak=system_peak,
    )


def deduplicate_recommendations(
    recommendations: List[DiagnosticRecommendation],
) -> List[DiagnosticRecommendation]:
    """Convenience: deduplicate recommendations."""
    return engine.deduplicate_recommendations(recommendations)
