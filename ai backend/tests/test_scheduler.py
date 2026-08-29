"""
tests/test_scheduler.py — Stage 14 scheduled AI runner (20 scenarios).
Deterministic: no Firebase. Runners/locks/state are injected.
"""
from __future__ import annotations

import datetime
import logging
import os
import threading
import time
import types

import pandas as pd
import pytest

from ai.scheduler import (Scheduler, SchedulerConfig, StateStore, JOB_COMPLETED,
                          JOB_FAILED, run_ai_pipeline)
from ai.report_generator import generate_monthly_report, demo_input


UTC = datetime.timezone.utc


def _now(y=2026, m=8, d=15, h=3):
    return datetime.datetime(y, m, d, h, 0, tzinfo=UTC).timestamp()


# ---------------------------------------------------------------------------
# 1. scheduler startup / shutdown logging
# ---------------------------------------------------------------------------

def test_scheduler_startup_and_shutdown_logs(caplog, tmp_path):
    caplog.set_level(logging.INFO, logger="ai.scheduler")
    cfg = SchedulerConfig(enabled=False, state_file=str(tmp_path / "s.json"))
    sched = Scheduler(config=cfg)
    ev = threading.Event()
    t = threading.Thread(target=sched.run_forever, kwargs=dict(poll_seconds=1, stop_event=ev))
    t.start()
    time.sleep(0.2)
    ev.set()
    t.join(timeout=3)
    assert any("started" in r.message for r in caplog.records)
    assert any("shutdown complete" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 2. scheduler disabled
# ---------------------------------------------------------------------------

def test_scheduler_disabled_skips(tmp_path):
    cfg = SchedulerConfig(enabled=False, state_file=str(tmp_path / "s.json"))
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0}
    sched = Scheduler(config=cfg, state_store=state,
                      ai_runner=lambda s, now: calls.update(n=calls["n"] + 1),
                      clock=lambda: 1000.0)
    sched.tick(1000.0)
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# 3. scheduled execution
# ---------------------------------------------------------------------------

def test_scheduled_execution_runs(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), ai_interval_seconds=3600)
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0}
    sched = Scheduler(config=cfg, state_store=state,
                      ai_runner=lambda s, now: calls.update(n=calls["n"] + 1),
                      has_new_data=lambda s, ts, now: True, clock=lambda: 1000.0)
    sched.tick(1000.0)
    assert calls["n"] == 1
    assert state.job("ai_processing").status == JOB_COMPLETED


# ---------------------------------------------------------------------------
# 4. duplicate-run prevention (interval gate)
# ---------------------------------------------------------------------------

def test_duplicate_run_prevention(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), ai_interval_seconds=3600)
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0}
    sched = Scheduler(config=cfg, state_store=state,
                      ai_runner=lambda s, now: calls.update(n=calls["n"] + 1),
                      has_new_data=lambda s, ts, now: True, clock=lambda: 1000.0)
    sched.tick(1000.0)   # runs, next_run = 4600
    sched.tick(1000.0)   # within interval -> skip
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 5. lock acquisition (another live instance)
# ---------------------------------------------------------------------------

def test_lock_acquisition_blocks_other_instance(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"))
    state = StateStore(str(tmp_path / "s.json"))
    j = state.job("ai_processing")
    j.locked_at = 1000.0
    j.locked_by = 999
    state.set_job("ai_processing", j)
    calls = {"n": 0}
    sched = Scheduler(config=cfg, state_store=state,
                      ai_runner=lambda s, now: calls.update(n=calls["n"] + 1),
                      clock=lambda: 1000.0)
    sched.tick(1000.0)
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# 6. lock recovery after timeout
# ---------------------------------------------------------------------------

def test_lock_recovery_after_timeout(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), lock_timeout_seconds=1800)
    state = StateStore(str(tmp_path / "s.json"))
    j = state.job("ai_processing")
    j.locked_at = 1000.0 - 9999  # long expired
    j.locked_by = 999
    state.set_job("ai_processing", j)
    calls = {"n": 0}
    sched = Scheduler(config=cfg, state_store=state,
                      ai_runner=lambda s, now: calls.update(n=calls["n"] + 1),
                      has_new_data=lambda s, ts, now: True, clock=lambda: 1000.0)
    sched.tick(1000.0)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 7. successful pipeline execution updates state
# ---------------------------------------------------------------------------

def test_successful_run_updates_last_success(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"))
    state = StateStore(str(tmp_path / "s.json"))
    sched = Scheduler(config=cfg, state_store=state,
                      ai_runner=lambda s, now: {"status": "ok"},
                      has_new_data=lambda s, ts, now: True, clock=lambda: 1234.0)
    sched.tick(1234.0)
    job = state.job("ai_processing")
    assert job.status == JOB_COMPLETED
    assert job.last_successful_ts == 1234.0
    assert job.last_error is None


# ---------------------------------------------------------------------------
# 8. stage failure isolation (run_ai_pipeline)
# ---------------------------------------------------------------------------

def test_stage_failure_isolation(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr("ai.preprocessing.run_preprocessing_pipeline",
                        lambda settings: {1: types.SimpleNamespace(
                            feature_frame=pd.DataFrame({"timestamp": [1, 2], "power": [1.0, 2.0]}))})
    monkeypatch.setattr("ai.anomaly_detection.run_anomaly_detection_pipeline",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("ai.persist_ai_results.run_stage_5_pipeline",
                        lambda *a, **k: calls.update(stage5=calls.get("stage5", 0) + 1) or {})
    monkeypatch.setattr("ai.peak_detection.run_peak_detection_pipeline",
                        lambda *a, **k: ({1: object()}, None))
    monkeypatch.setattr("ai.maintenance_risk.run_maintenance_risk_pipeline",
                        lambda *a, **k: ({1: object()}, None))
    monkeypatch.setattr("ai.forecast.run_forecast_pipeline",
                        lambda *a, **k: ({1: object()}, None))
    monkeypatch.setattr("ai.bill_prediction.compute_actual_energy_from_history", lambda *a, **k: 0.0)
    monkeypatch.setattr("ai.bill_prediction.predict_bill_from_record",
                        lambda *a, **k: {"status": "OK", "anchor_timestamp": 1})
    monkeypatch.setattr("ai.bill_prediction.write_bill_prediction", lambda *a, **k: True)
    monkeypatch.setattr("ai.energy_saving.run_stage_11_pipeline",
                        lambda *a, **k: calls.update(stage11=calls.get("stage11", 0) + 1) or {"recommendations": []})

    settings = types.SimpleNamespace(pzem_count=1)
    # must not raise despite the anomaly stage failing
    run_ai_pipeline(settings)
    assert calls.get("stage5") == 1
    assert calls.get("stage11") == 1


# ---------------------------------------------------------------------------
# 9-11. retry behaviour
# ---------------------------------------------------------------------------

def test_transient_failure_retried_then_success(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), retry_count=2, retry_delay_seconds=0)
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0}
    def runner(s, now):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return {"ok": True}
    sched = Scheduler(config=cfg, state_store=state, ai_runner=runner,
                      has_new_data=lambda s, ts, now: True, clock=lambda: 1000.0)
    sched.tick(1000.0)
    assert calls["n"] == 2
    assert state.job("ai_processing").status == JOB_COMPLETED


def test_retry_limit_reached_marks_failed(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), retry_count=2, retry_delay_seconds=0)
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0}
    def runner(s, now):
        calls["n"] += 1
        raise RuntimeError("always")
    sched = Scheduler(config=cfg, state_store=state, ai_runner=runner,
                      has_new_data=lambda s, ts, now: True, clock=lambda: 1000.0)
    sched.tick(1000.0)
    assert calls["n"] == 3  # initial + 2 retries
    assert state.job("ai_processing").status == JOB_FAILED


# ---------------------------------------------------------------------------
# 12. restart recovery (stale lock from a crashed pid)
# ---------------------------------------------------------------------------

def test_restart_recovery_runs_stale_lock(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), lock_timeout_seconds=1800)
    state = StateStore(str(tmp_path / "s.json"))
    j = state.job("ai_processing")
    j.locked_at = 1000.0 - 99999
    j.locked_by = 4242  # previous (dead) pid
    state.set_job("ai_processing", j)
    calls = {"n": 0}
    sched = Scheduler(config=cfg, state_store=state,
                      ai_runner=lambda s, now: calls.update(n=calls["n"] + 1),
                      has_new_data=lambda s, ts, now: True, clock=lambda: 1000.0)
    sched.tick(1000.0)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 13. last-success tracking drives incremental skip
# ---------------------------------------------------------------------------

def test_last_success_tracking(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), ai_interval_seconds=1)
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0}
    sched = Scheduler(config=cfg, state_store=state,
                      ai_runner=lambda s, now: calls.update(n=calls["n"] + 1),
                      has_new_data=lambda s, ts, now: False, clock=lambda: 1000.0)
    sched.tick(1000.0)  # bootstrap (last_successful_ts None) -> runs
    sched.tick(2000.0)  # last_successful_ts set + no new data -> skip
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 14. incremental processing guard
# ---------------------------------------------------------------------------

def test_incremental_no_new_data_skips(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), ai_interval_seconds=1)
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0}
    sched = Scheduler(config=cfg, state_store=state,
                      ai_runner=lambda s, now: calls.update(n=calls["n"] + 1),
                      has_new_data=lambda s, ts, now: False, clock=lambda: 1000.0)
    sched.tick(1000.0)  # bootstrap run
    sched.tick(5000.0)  # within logic, no new data -> skip
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 15. monthly report trigger (completed month)
# ---------------------------------------------------------------------------

def test_monthly_report_trigger(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), monthly_report_hour=2)
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0, "args": None}
    def runner(s, now, y, m):
        calls["n"] += 1
        calls["args"] = (y, m)
    now = _now(2026, 9, 1, 3)
    sched = Scheduler(config=cfg, state_store=state, monthly_runner=runner,
                      ai_runner=lambda s, now: None, clock=lambda: now)
    sched.tick(now)
    assert calls["n"] == 1
    assert calls["args"] == (2026, 8)
    assert state.job("monthly_report").last_monthly_report == "2026-08"


# ---------------------------------------------------------------------------
# 16. duplicate monthly report prevention
# ---------------------------------------------------------------------------

def test_monthly_report_no_duplicate(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), monthly_report_hour=2)
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0}
    runner = lambda s, now, y, m: calls.update(n=calls["n"] + 1)
    now = _now(2026, 9, 1, 3)
    sched = Scheduler(config=cfg, state_store=state, monthly_runner=runner,
                      ai_runner=lambda s, now: None, clock=lambda: now)
    sched.tick(now)
    sched.tick(now)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 17. incomplete current month handling
# ---------------------------------------------------------------------------

def test_incomplete_current_month_skipped(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"))
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0}
    runner = lambda s, now, y, m: calls.update(n=calls["n"] + 1)
    now = _now(2026, 8, 15, 3)
    sched = Scheduler(config=cfg, state_store=state, monthly_runner=runner,
                      ai_runner=lambda s, now: None, clock=lambda: now)
    sched.tick(now)
    assert calls["n"] == 0


def test_current_month_allowed_when_configured(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), monthly_report_allow_current=True)
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0, "args": None}
    def runner(s, now, y, m):
        calls["n"] += 1
        calls["args"] = (y, m)
    now = _now(2026, 8, 15, 3)
    sched = Scheduler(config=cfg, state_store=state, monthly_runner=runner,
                      ai_runner=lambda s, now: None, clock=lambda: now)
    sched.tick(now)
    assert calls["n"] == 1
    assert calls["args"] == (2026, 8)


# ---------------------------------------------------------------------------
# 18. deterministic monthly filename / target mapping
# ---------------------------------------------------------------------------

def test_monthly_target_mapping():
    cfg = SchedulerConfig()
    assert Scheduler.monthly_target(_now(2026, 9, 1, 3), cfg) == (2026, 8)
    assert Scheduler.monthly_target(_now(2026, 8, 15, 3), cfg) is None
    assert Scheduler.monthly_target(_now(2026, 1, 1, 3), cfg) == (2025, 12)
    cfg2 = SchedulerConfig(monthly_report_allow_current=True)
    assert Scheduler.monthly_target(_now(2026, 8, 15, 3), cfg2) == (2026, 8)


def test_monthly_report_filename_deterministic(tmp_path):
    res = generate_monthly_report(data=demo_input(), year=2026, month=8, output_dir=str(tmp_path))
    assert res["stub"] == "report-2026-08"
    assert os.path.exists(res["pdf"])


# ---------------------------------------------------------------------------
# 19. no unnecessary Firebase writes / idempotent persistence
# ---------------------------------------------------------------------------

class _Child:
    def __init__(self, db, path, key):
        self.db, self.path, self.key = db, path, key
    def get(self):
        return self.db.store.get((self.path, self.key))
    def set(self, val):
        self.db.store[(self.path, self.key)] = val
        self.db.sets += 1


class _Ref:
    def __init__(self, db, path):
        self.db, self.path = db, path
    def child(self, key):
        return _Child(self.db, self.path, key)


class _FakeDB:
    def __init__(self):
        self.store = {}
        self.sets = 0
        self.paths = []
    def reference(self, path):
        self.paths.append(path)
        return _Ref(self, path)


def test_idempotent_bill_write_no_duplicate(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr("ai.bill_prediction._db_ref", lambda path: _Ref(db, path))
    from ai import bill_prediction
    pred = {"status": "OK", "anchor_timestamp": 77}
    bill_prediction.write_bill_prediction(pred, 77)
    bill_prediction.write_bill_prediction(pred, 77)
    assert db.sets == 1  # second write is skipped (idempotent)


def test_idempotent_energy_saving_write_no_duplicate(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr("ai.energy_saving._db_ref", lambda path: _Ref(db, path))
    from ai import energy_saving
    energy_saving.write_energy_saving([], 99)
    energy_saving.write_energy_saving([], 99)
    assert db.sets == 1


def test_reports_not_written_to_rtdb(tmp_path, monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr("ai.bill_prediction._db_ref", lambda path: _Ref(db, path))
    monkeypatch.setattr("ai.energy_saving._db_ref", lambda path: _Ref(db, path))
    res = generate_monthly_report(data=demo_input(), year=2026, month=8, output_dir=str(tmp_path))
    assert os.path.exists(res["pdf"])
    # the monthly PDF never touches Firebase RTDB (local disk only)
    assert not any("reports" in p for p in db.paths)


def test_no_writes_when_no_new_data(tmp_path):
    cfg = SchedulerConfig(state_file=str(tmp_path / "s.json"), ai_interval_seconds=1)
    state = StateStore(str(tmp_path / "s.json"))
    calls = {"n": 0}
    sched = Scheduler(config=cfg, state_store=state,
                      ai_runner=lambda s, now: calls.update(n=calls["n"] + 1),
                      has_new_data=lambda s, ts, now: False, clock=lambda: 1000.0)
    sched.tick(1000.0)
    sched.tick(5000.0)
    assert calls["n"] == 1  # only the bootstrap run, then incremental skip
