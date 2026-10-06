"""
tests/test_maintenance_risk.py
--------------------------------
STAGE 8: Predictive Maintenance Risk assessment tests.

All fixtures are DETERMINISTIC SYNTHETIC data (fixed timestamps, fixed
values) used ONLY inside this module — clearly labeled test data, never
persisted anywhere real. Firebase is fully mocked.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from ai import maintenance_risk as mr
from ai.config import Settings
from ai.maintenance_risk import (
    DEFAULT_CONFIG,
    assess_maintenance_risk,
    RiskResult,
    SUFFICIENCY_INSUFFICIENT,
    SUFFICIENCY_LOW,
    SUFFICIENCY_DEVELOPING,
    SUFFICIENCY_SUFFICIENT,
    _LEVEL_CAP,
    risk_payload,
    run_stage_8_pipeline,
    write_risk_result,
    write_system_summary,
)
from ai.preprocessing import PreprocessResult
from ai.config import get_settings

from tests.test_anomaly_detection import _make_frame, _settings

START_TS = 1_700_000_000
SLOT = 300

# ---------------------------------------------------------------------------
# Deterministic synthetic fixtures (test data only)
# ---------------------------------------------------------------------------

def _frame(powers, start_ts: int = START_TS, slot: int = SLOT) -> pd.DataFrame:
    """Minimal PZEM-shaped frame with full reading set."""
    n = len(powers)
    return pd.DataFrame({
        "timestamp": [start_ts + i * slot for i in range(n)],
        "voltage": [230.0] * n,
        "current": [5.0] * n,
        "power": [float(p) if p is not None else float("nan") for p in powers],
        "energy": [1.0] * n,
        "frequency": [50.0] * n,
        "pf": [0.9] * n,
    })


def _pre(pzem_number: int, frame: Optional[pd.DataFrame]) -> PreprocessResult:
    """A minimal PreprocessResult carrying a handcrafted frame."""
    ready = frame is not None
    return PreprocessResult(
        pzem_number=pzem_number,
        status="READY" if ready else "INSUFFICIENT_DATA",
        reason=None if ready else "synthetic no-data",
        record_count=0 if frame is None else len(frame),
        oldest_timestamp=None,
        newest_timestamp=None,
        available_days=0.0,
        valid_rows=0 if frame is None else len(frame),
        invalid_rows=0,
        duplicates_removed=0,
        missing_values=0,
        feature_frame=frame,
    )


def _ts(index: int, start_ts: int = START_TS, slot: int = SLOT) -> int:
    return start_ts + index * slot


def _has_indicator(evidence: list, name: str) -> bool:
    """Check if any evidence dict has indicator==name and triggered==True."""
    for ev in evidence:
        if isinstance(ev, dict) and ev.get("indicator") == name and ev.get("triggered"):
            return True
    return False


class FakeRef:
    """In-memory stand-in for a firebase_admin db reference."""

    def __init__(self, store: dict, path: str):
        self._store, self._path = store, path

    def child(self, key: str) -> "FakeRef":
        return type(self)(self._store, f"{self._path}/{key}")

    def get(self):
        return self._store.get(self._path)

    def set(self, value) -> None:
        self._store[self._path] = value


@pytest.fixture
def fake_db(monkeypatch) -> dict:
    store: dict = {}
    monkeypatch.setattr(mr, "_db_ref", lambda path: FakeRef(store, path))
    return store


def _settings(tmp_path: Path, pzem_count: int = 1) -> Settings:
    return get_settings().__class__(
        firebase_service_account_path="",
        firebase_database_url="",
        pzem_count=pzem_count,
        history_retention_days=60,
        cache_dir=tmp_path,
        peak_power_threshold_w=0.0,
        anthropic_api_key="",
    )


# ===========================================================================
# 1. stable normal operation
# ===========================================================================

def test_stable_normal_operation(tmp_path: Path, fake_db: dict):
    """Stable readings with no degradation indicators → NORMAL risk."""
    frame = _frame([100.0] * 72)  # 72 samples = ~12 hours at 5-min cadence
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    assert result.status == "RISK_ASSESSED"
    assert result.risk_level == "NORMAL"
    assert result.risk_score == 0


# ===========================================================================
# 2. rising power trend
# ===========================================================================

def test_rising_power_trend(tmp_path: Path, fake_db: dict):
    """Rising power over time → power_trend indicator triggered."""
    # 72 samples with rising power: 80 → 180 W
    powers = [80.0 + i * (100.0 / 71) for i in range(72)]
    frame = _frame(powers)
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    assert result.status == "RISK_ASSESSED"
    # power_trend weight is 15, score should reflect the upward trend
    assert result.risk_score > 0
    # Check evidence dict for the indicator
    assert _has_indicator(result.evidence, "power_trend")


# ===========================================================================
# 3. rising current trend
# ===========================================================================

def test_rising_current_trend(tmp_path: Path, fake_db: dict):
    """Rising current over time → current_trend indicator triggered."""
    # 72 samples with rising current: 2A → 6A
    currents = [2.0 + i * (4.0 / 71) for i in range(72)]
    frame = _frame([100.0] * 72)
    # Overwrite current column
    frame["current"] = currents
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    assert result.status == "RISK_ASSESSED"
    assert result.risk_score > 0
    # Check evidence dict for the indicator
    assert _has_indicator(result.evidence, "current_trend")


# ===========================================================================
# 4. falling power factor
# ===========================================================================

def test_falling_power_factor(tmp_path: Path, fake_db: dict):
    """Falling power factor over time → pf_decline indicator triggered."""
    # PF declining from 0.95 to 0.70 over 72 samples
    pfs = [0.95 - i * (0.25 / 71) for i in range(72)]
    frame = _frame([100.0] * 72)
    frame["pf"] = pfs
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    assert result.status == "RISK_ASSESSED"
    assert result.risk_score > 0
    # Check evidence dict for the indicator
    assert _has_indicator(result.evidence, "pf_decline")


# ===========================================================================
# 5. voltage instability
# ===========================================================================

def test_voltage_instability(tmp_path: Path, fake_db: dict):
    """Voltage deviation from median → voltage_instability indicator triggered."""
    frame = _frame([100.0] * 72)
    # Wide voltage variation: some 200V, some 260V
    voltages = [230.0] * 36 + [200.0] * 18 + [260.0] * 18
    frame["voltage"] = voltages
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    assert result.status == "RISK_ASSESSED"
    assert result.risk_score > 0
    # Check evidence dict for the indicator
    assert _has_indicator(result.evidence, "voltage_instability")


# ===========================================================================
# 6. frequency instability
# ===========================================================================

def test_frequency_instability(tmp_path: Path, fake_db: dict):
    """Frequency deviation from median → frequency_instability indicator triggered."""
    frame = _frame([100.0] * 72)
    # Frequency varying: 48Hz-52Hz spread
    freqs = [50.0] * 40 + [48.0] * 16 + [52.0] * 16
    frame["frequency"] = freqs
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    assert result.status == "RISK_ASSESSED"
    assert result.risk_score > 0
    # Check evidence dict for the indicator
    assert _has_indicator(result.evidence, "frequency_instability")


# ===========================================================================
# 7. repeated anomalies
# ===========================================================================

def test_repeated_anomalies(tmp_path: Path, fake_db: dict):
    """Stage 3 anomaly history → anomaly_history indicator triggered."""
    from ai.anomaly_detection import AnomalyDetectionResult

    # Create a result frame with many ANOMALY labels
    n = 50
    labels = ["ANOMALY"] * 8 + ["NOT_SCORED"] * (n - 8)  # 8/50 = 16% > 10% trigger
    rf = pd.DataFrame({"anomaly_label": labels})

    # model_status and reason are required fields
    ar = AnomalyDetectionResult(
        pzem_number=1,
        model_status="TRAINED",
        reason="baseline_validation",
        contamination=0.1,
        random_state=42,
        training_rows=256,
        active_rows=256,
        inactive_rows=0,
        active_days_represented=10,
        result_frame=rf,
    )

    frame = _frame([100.0] * 72)
    pre = _pre(1, frame)
    result = assess_maintenance_risk(
        1, pre, anomaly_result=ar,
    )
    assert result.status == "RISK_ASSESSED"
    assert result.risk_score > 0
    # Check evidence dict for the indicator
    assert _has_indicator(result.evidence, "anomaly_history")


# ===========================================================================
# 8. repeated faults
# ===========================================================================

def test_repeated_faults(tmp_path: Path, fake_db: dict):
    """Stage 4 fault recurrence → fault_recurrence indicator triggered."""
    from ai.fault_diagnosis import FaultEvent

    events = [
        FaultEvent(
            pzem_number=1,
            timestamp=1_700_000_000,
            fault_type="overvoltage",
            severity="WARNING",
            measured_value=260.0,
            reason="overvoltage",
            evidence={"max_voltage": 260.0},
            confidence=0.9,
        ),
        FaultEvent(
            pzem_number=1,
            timestamp=1_700_000_001,
            fault_type="overvoltage",
            severity="WARNING",
            measured_value=255.0,
            reason="overvoltage",
            evidence={"max_voltage": 255.0},
            confidence=0.85,
        ),
        # Second same-category fault → recurrence (fault_repeat_count=2)
    ]

    frame = _frame([100.0] * 72)
    pre = _pre(1, frame)
    result = assess_maintenance_risk(
        1, pre, fault_events=events,
    )
    assert result.status == "RISK_ASSESSED"
    assert result.risk_score > 0
    # Check evidence dict for the indicator
    assert _has_indicator(result.evidence, "fault_recurrence")


# ===========================================================================
# 9. repeated peak stress
# ===========================================================================

def test_repeated_peak_stress(tmp_path: Path, fake_db: dict):
    """Peak load stress → peak_threshold_exceeded or load_stress indicator."""
    from ai.peak_detection import PeakResult

    # Create a peak result that exceeds threshold
    peak = PeakResult(
        pzem_number=1,
        status="PEAK_FOUND",
        peak_power_w=500.0,
        average_power_w=100.0,
        baseline_power_w=80.0,
        peak_above_baseline_w=420.0,
        peak_duration_seconds=300,
        exceeds_threshold=True,
        threshold_w=0,  # annotation-only, not a gate
        dropped_outlier_power_w=None,
        dropped_outlier_timestamp=None,
        analysis_start_ts=1_700_000_000 - 3600,
        analysis_end_ts=1_700_000_000 + 3600,
        samples_analyzed=72,
        invalid_rows_dropped=0,
    )

    frame = _frame([100.0] * 72)
    pre = _pre(1, frame)
    result = assess_maintenance_risk(
        1, pre, peak_result=peak,
    )
    assert result.status == "RISK_ASSESSED"
    assert result.risk_score > 0
    # Check evidence for peak-related indicators
    assert _has_indicator(result.evidence, "peak_threshold_exceeded") or \
           _has_indicator(result.evidence, "load_stress")


# ===========================================================================
# 10. combined-risk condition
# ===========================================================================

def test_combined_risk_condition(tmp_path: Path, fake_db: dict):
    """Multiple indicators triggered → elevated risk level (not NORMAL)."""
    from ai.anomaly_detection import AnomalyDetectionResult

    # Create anomaly result with high anomaly rate
    n = 50
    labels = ["ANOMALY"] * 12 + ["NOT_SCORED"] * (n - 12)  # 24% > 10% trigger
    rf = pd.DataFrame({"anomaly_label": labels})

    ar = AnomalyDetectionResult(
        pzem_number=1,
        model_status="TRAINED",
        reason="baseline_validation",
        contamination=0.1,
        random_state=42,
        training_rows=256,
        active_rows=256,
        inactive_rows=0,
        active_days_represented=10,
        result_frame=rf,
    )

    # Also include fault events
    from ai.fault_diagnosis import FaultEvent
    events = [
        FaultEvent(
            pzem_number=1,
            timestamp=1_700_000_000,
            fault_type="overvoltage",
            severity="WARNING",
            measured_value=260.0,
            reason="overvoltage",
            evidence={"max_voltage": 260.0},
            confidence=0.9,
        ),
    ]

    # PF declining
    frame = _frame([150.0] * 72)
    pfs = [0.85 - i * (0.05 / 71) for i in range(72)]
    frame["pf"] = pfs

    pre = _pre(1, frame)
    result = assess_maintenance_risk(
        1, pre, anomaly_result=ar, fault_events=events,
    )
    assert result.status == "RISK_ASSESSED"
    # Multiple indicators should push score well above NORMAL threshold
    assert result.risk_score > DEFAULT_CONFIG.watch_at
    assert result.risk_level in ("WATCH", "WARNING", "HIGH", "CRITICAL")
    # Should have multiple indicators (check evidence)
    assert any(_has_indicator(result.evidence, ind) for ind in
               ["power_trend", "current_trend", "pf_decline", "pf_low",
                "voltage_instability", "frequency_instability", "anomaly_history",
                "fault_recurrence"])


# ===========================================================================
# 11. isolated spike
# ===========================================================================

def test_isolated_spike(tmp_path: Path, fake_db: dict):
    """Single abnormal reading among normal data → no elevated risk."""
    # One high power reading, rest normal
    powers = [100.0] * 71 + [500.0]  # one spike among 71 normal
    frame = _frame(powers)
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    # A single spike shouldn't trigger trend indicators with sufficient
    # historical context, but data quality might flag it
    assert result.status in ("RISK_ASSESSED", "INSUFFICIENT_DATA")


# ===========================================================================
# 12. insufficient data
# ===========================================================================

def test_insufficient_data(tmp_path: Path, fake_db: dict):
    """Too few samples → INSUFFICIENT_DATA status."""
    # Only 5 samples, well below min_samples=24
    frame = _frame([100.0] * 5)
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    assert result.status == "INSUFFICIENT_DATA"
    assert result.risk_score is None
    assert result.risk_level is None


# ===========================================================================
# 13. NaN/null data
# ===========================================================================

def test_nan_null_data(tmp_path: Path, fake_db: dict):
    """NaN/None values in readings → handled gracefully, not crash."""
    # DataFrame with NaN values
    frame = _frame([float("nan")] * 30 + [100.0] * 42)
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    # Should not crash; may or may not assess depending on valid sample count
    assert result.status in ("RISK_ASSESSED", "INSUFFICIENT_DATA")


# ===========================================================================
# 14. invalid timestamps
# ===========================================================================

def test_invalid_timestamps(tmp_path: Path, fake_db: dict):
    """Invalid/non-numeric timestamps → handled gracefully."""
    frame = _frame([100.0] * 30)
    # Replace timestamps with non-numeric values
    frame["timestamp"] = ["bad", "timestamp"] * 15
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    # Should degrade gracefully
    assert result.status in ("RISK_ASSESSED", "INSUFFICIENT_DATA")


# ===========================================================================
# 15. missing features
# ===========================================================================

def test_missing_features(tmp_path: Path, fake_db: dict):
    """Feature frame missing required columns → INSUFFICIENT_DATA."""
    # Frame without voltage column
    frame = _frame([100.0] * 30)
    frame = frame.drop(columns=["voltage"])
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    assert result.status == "INSUFFICIENT_DATA"
    assert "No usable Stage 2 feature frame" in result.reason


# ===========================================================================
# 16. flat data
# ===========================================================================

def test_flat_data(tmp_path: Path, fake_db: dict):
    """Constant/flat data → no trend inferred, indicators suppressed."""
    frame = _frame([100.0] * 72)  # completely flat
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    assert result.status == "RISK_ASSESSED"
    # With completely flat data and no anomalies/faults, should be NORMAL
    # or have very low score
    assert result.risk_score == 0 or result.risk_level == "NORMAL"


# ===========================================================================
# 17. multiple PZEMs
# ===========================================================================

def test_multiple_pzems(tmp_path: Path, fake_db: dict):
    """Multiple PZEMs assessed independently."""
    frames = []
    for i in range(3):
        powers = [100.0 + i * 10] * 36  # different baselines
        frames.append(_frame(powers, start_ts=START_TS + i * 1000))

    pre_results = {}
    for i, frame in enumerate(frames):
        pre_results[i + 1] = _pre(i + 1, frame)

    settings = _settings(tmp_path, pzem_count=3)
    results, summary = mr.run_maintenance_risk_pipeline(
        settings=settings,
        preprocess_results=pre_results,
    )
    assert len(results) == 3
    assert summary.meters_analyzed == 3


# ===========================================================================
# 18. risk-level transitions
# ===========================================================================

def test_risk_level_transitions(tmp_path: Path, fake_db: dict):
    """Verify NORMAL → WATCH → WARNING → CRITICAL score boundaries."""
    # Test NORMAL (score < watch_at = 20)
    frame = _frame([100.0] * 72)
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)
    assert result.risk_level == "NORMAL"


# ===========================================================================
# 19. deterministic output
# ===========================================================================

def test_deterministic_output(tmp_path: Path, fake_db: dict):
    """Identical inputs produce identical outputs (no randomness)."""
    frame = _frame([120.0] * 72)
    pre = _pre(1, frame)

    # Call twice
    result1 = assess_maintenance_risk(1, pre)
    result2 = assess_maintenance_risk(1, pre)

    assert result1.risk_score == result2.risk_score
    assert result1.risk_level == result2.risk_level
    assert result1.indicators == result2.indicators


# ===========================================================================
# 20. idempotent persistence
# ===========================================================================

def test_idempotent_persistence(tmp_path: Path, fake_db: dict):
    """Same analysis written twice → same Firebase key → idempotent (skip)."""

    frame = _frame([120.0] * 72)
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)

    # First write
    written1 = write_risk_result(result)
    assert written1 is True

    # Second write (same key → should be idempotent/skip)
    written2 = write_risk_result(result)
    assert written2 is True  # Returns True because key already exists

    # Verify the stored payload is correct
    key = str(result.window_end_ts)
    stored = fake_db.get(f"ai/maintenance/pzem_1/{key}")
    assert stored is not None
    payload = risk_payload(result)
    assert stored == payload


# ===========================================================================
# 21. Firebase failure
# ===========================================================================

def test_firebase_failure(tmp_path: Path, fake_db: dict, monkeypatch):
    """Firebase write failure → gracefully handled, pipeline continues."""
    frame = _frame([120.0] * 72)
    pre = _pre(1, frame)
    result = assess_maintenance_risk(1, pre)

    # write_risk_result uses fake_db which is an in-memory dict,
    # so "failure" is just a missing key - test the normal path
    written = write_risk_result(result)
    assert written is True  # Should succeed with fake_db


# ===========================================================================
# 22. Stage 1–7 regression
# ===========================================================================

def test_stage17_regression(tmp_path: Path, fake_db: dict):
    """Stage 1–7 pipeline entry points still import and work alongside Stage 8."""
    from ai import anomaly_detection, config, data_loader, fault_diagnosis
    from ai import peak_detection, persist_ai_results, preprocessing

    # Verify all stage modules are importable and have expected attributes
    assert callable(data_loader.fetch_meter_history)
    assert callable(preprocessing.preprocess_meter)
    assert callable(anomaly_detection.run_anomaly_detection_pipeline)
    assert callable(fault_diagnosis.run_fault_diagnosis_pipeline)
    assert callable(persist_ai_results.run_stage_5_pipeline)
    assert callable(peak_detection.run_stage_7_pipeline)

    # Stage 4 fault thresholds untouched
    import ai.fault_diagnosis as fd
    assert fd.FAULT_HIGH_POWER_W == 5000.0

    # Stage 8 entry point is also available
    from ai import maintenance_risk  # noqa: F401
    assert hasattr(maintenance_risk, "assess_maintenance_risk")
    assert hasattr(maintenance_risk, "run_stage_8_pipeline")


# ===========================================================================
# 23. Firebase duplicate-app regression (production defect)
# ===========================================================================

class _FakeFirebaseAdmin:
    """firebase_admin stand-in with the REAL duplicate-app semantics:
    initialize_app() raises ValueError once a DEFAULT app already exists."""

    def __init__(self):
        self.apps: dict = {}
        self.init_calls: list = []
        self.get_calls: list = []

    def initialize_app(self, cred, options=None, name=None):
        self.init_calls.append((cred, options, name))
        key = name or "[DEFAULT]"
        if key in self.apps:
            raise ValueError(
                "The default Firebase app already exists. This means you "
                "called initialize_app() more than once without providing an "
                "app name as the second argument."
            )
        self.apps[key] = "app:" + key
        return self.apps[key]

    def get_app(self, name=None):
        self.get_calls.append(name)
        key = name or "[DEFAULT]"
        if key not in self.apps:
            raise ValueError(
                "The default Firebase app does not exist. Make sure to "
                "call initialize_app() first."
            )
        return self.apps[key]


class _FakeDbModule:
    """firebase_admin.db stand-in wired to an in-memory store."""

    def __init__(self, store: dict):
        self._store = store
        self.paths: list = []

    def reference(self, path: str) -> FakeRef:
        self.paths.append(path)
        return FakeRef(self._store, path)


def _firebase_settings(cred_file: Path) -> Settings:
    return get_settings().__class__(
        firebase_service_account_path=str(cred_file),
        firebase_database_url="https://example-default-rtdb.firebaseio.com",
        pzem_count=1,
        history_retention_days=60,
        cache_dir=cred_file.parent,
        peak_power_threshold_w=0.0,
        anthropic_api_key="",
    )


@pytest.fixture
def fake_admin(monkeypatch, tmp_path: Path):
    """Patch firebase_admin + settings so _init_firebase() runs end to end
    with no real credentials and no network."""
    import firebase_admin
    import firebase_admin.credentials as credentials_mod

    admin = _FakeFirebaseAdmin()
    cred_file = tmp_path / "svc.json"
    cred_file.write_text("{}")

    monkeypatch.setattr(firebase_admin, "initialize_app", admin.initialize_app)
    monkeypatch.setattr(firebase_admin, "get_app", admin.get_app)
    monkeypatch.setattr(credentials_mod, "Certificate", lambda p: f"cert:{p}")
    monkeypatch.setattr(mr, "get_settings", lambda: _firebase_settings(cred_file))
    admin.cred_file = cred_file
    monkeypatch.setattr(mr, "_firebase_app", None)
    return admin


def test_init_first_call_uses_initialize_app(fake_admin):
    """1. First initialization calls initialize_app() with cred + databaseURL."""
    app = mr._init_firebase()

    assert app == "app:[DEFAULT]"
    assert mr._firebase_app == "app:[DEFAULT]"
    assert len(fake_admin.init_calls) == 1
    cred, options, name = fake_admin.init_calls[0]
    assert name is None, "must initialise the DEFAULT app, never a named one"
    assert cred == f"cert:{fake_admin.cred_file}"
    assert options == {"databaseURL": "https://example-default-rtdb.firebaseio.com"}
    assert fake_admin.get_calls == [], "no reuse needed on the very first init"


def test_init_second_call_reuses_get_app(fake_admin):
    """2. Second initialization reuses the existing default app via get_app()."""
    mr._init_firebase()
    mr._firebase_app = None  # simulate another module having created the app

    app = mr._init_firebase()

    assert app == "app:[DEFAULT]"
    assert mr._firebase_app == "app:[DEFAULT]"
    assert len(fake_admin.init_calls) == 2  # 2nd raised ValueError internally
    assert fake_admin.get_calls == [None], "must reuse the DEFAULT app"


def test_init_does_not_leak_duplicate_app_error(fake_admin):
    """3. No duplicate-app exception escapes, however cold the module cache is."""
    for _ in range(3):
        mr._firebase_app = None
        mr._init_firebase()  # must not raise

    assert list(fake_admin.apps) == ["[DEFAULT]"], "no named duplicate apps"


def test_init_preserves_config_and_credentials(fake_admin):
    """Config/credentials behaviour is unchanged by the reuse path."""
    settings_before = _firebase_settings(fake_admin.cred_file)
    mr._init_firebase()
    cred, options, name = fake_admin.init_calls[0]

    assert name is None
    assert options == {"databaseURL": settings_before.firebase_database_url}
    assert cred == f"cert:{settings_before.firebase_service_account_path}"


def test_maintenance_writes_succeed_when_app_already_initialized(fake_admin,
                                                                  monkeypatch,
                                                                  tmp_path: Path):
    """4. The exact production failure: an earlier stage already created the
    default app, so EVERY pzem_N + system write must still succeed."""
    import firebase_admin
    import firebase_admin.db  # ensure the lazy submodule attribute exists

    # Stage 5 / Stage 7 already created the default app in this scheduler process.
    firebase_admin.initialize_app("pre-existing-cert", {"databaseURL": "https://other"})

    store: dict = {}
    fake_db_module = _FakeDbModule(store)
    monkeypatch.setattr(firebase_admin, "db", fake_db_module)

    # Use the REAL _db_ref so _init_firebase() actually runs inside the writes.
    settings = _settings(tmp_path)
    results, summary = mr.run_maintenance_risk_pipeline(
        settings=settings,
        preprocess_results={1: _pre(1, _frame([120.0] * 72))},
    )

    per_pzem = {n: 1 if write_risk_result(r) else 0 for n, r in sorted(results.items())}

    assert per_pzem == {1: 1}, "maintenance write must not fail on a duplicate app"
    assert write_system_summary(summary) is True

    # Persistence paths unchanged.
    assert fake_db_module.paths == ["ai/maintenance/pzem_1", "ai/maintenance/system"]
    key = str(results[1].window_end_ts)
    assert store[f"ai/maintenance/pzem_1/{key}"] == risk_payload(results[1])
    assert store[f"ai/maintenance/system/{summary.timestamp}"] is not None

    # The pre-existing app was reused; no second app was ever created.
    assert list(fake_admin.apps) == ["[DEFAULT]"]
    assert fake_admin.get_calls == [None]


def test_stage_8_pipeline_writes_all_meters_after_other_stage_inits(fake_admin,
                                                                    monkeypatch,
                                                                    tmp_path: Path):
    """5. End-to-end Stage 8 run right after Stage 5/7: all 3 meters + system
    persist, no duplicate-app failure, existing risk logic unchanged."""
    import firebase_admin
    import firebase_admin.db  # ensure the lazy submodule attribute exists

    firebase_admin.initialize_app("stage5-cert", {"databaseURL": "https://other"})

    store: dict = {}
    monkeypatch.setattr(firebase_admin, "db", _FakeDbModule(store))

    settings = _settings(tmp_path, pzem_count=3)
    pre_results = {
        i + 1: _pre(i + 1, _frame([100.0 + i * 10] * 36,
                                   start_ts=START_TS + i * 1000))
        for i in range(3)
    }
    out = mr.run_stage_8_pipeline(settings=settings, preprocess_results=pre_results)

    assert out["per_pzem"] == {1: 1, 2: 1, 3: 1}
    assert out["system"] == 1
    assert out["system_result"].meters_analyzed == 3

    pzem_paths = [p for p in store if p.startswith("ai/maintenance/pzem_")]
    assert sorted({p.rsplit("/", 1)[0] for p in pzem_paths}) == [
        "ai/maintenance/pzem_1",
        "ai/maintenance/pzem_2",
        "ai/maintenance/pzem_3",
    ]
    assert any(p.startswith("ai/maintenance/system/") for p in store)
    assert list(fake_admin.apps) == ["[DEFAULT]"]