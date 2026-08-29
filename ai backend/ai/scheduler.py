"""
ai/scheduler.py
---------------
STAGE 14: Storage-Efficient Scheduled AI Runner.

Orchestrates the EXISTING Stage 1-13 functions on a schedule. It introduces NO
new AI algorithms and NO duplicate pipeline: it only calls the same functions the
standalone stage scripts already call.

Design (storage-safe by construction):
  * Every downstream write (anomalies, faults, peaks, risk, forecast, bill,
    energy-saving) uses the existing idempotent get-then-set persistence
    (keyed by record timestamp / anchor). Re-running the pipeline therefore
    never creates duplicate records in /ai/*.
  * Stage 1 data fetch (data_loader.fetch_meter_history) is already incremental
    (only slots newer than the local cache are pulled from Firebase).
  * The scheduler only runs the AI pipeline when (a) the interval elapsed AND
    (b) there is actually new data since the last successful run, so we do not
    reprocess 60 days of history on every cycle when nothing changed.
  * The monthly PDF is generated at most once per completed month, and the PDF
    is written to local disk only — never to Firebase RTDB.
  * Job/lock state lives in a local JSON file (not Firebase), with a timeout so
    a crashed process cannot leave a permanent lock.

All durations/intervals are configurable via environment variables.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import signal
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger("ai.scheduler")

# Job status vocabulary
JOB_IDLE = "idle"
JOB_RUNNING = "running"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"


# ---------------------------------------------------------------------------
# Configuration (env-driven; no secrets)
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    enabled: bool = True
    ai_interval_seconds: int = 21600          # 6h
    monthly_report_hour: int = 2             # generate on day-1 at/after this UTC hour
    monthly_report_allow_current: bool = False
    retry_count: int = 3
    retry_delay_seconds: int = 30
    lock_timeout_seconds: int = 1800          # 30 min
    log_level: str = "INFO"
    state_file: str = ".scheduler_state.json"


def get_scheduler_config() -> SchedulerConfig:
    def _bool(name: str, default: bool) -> bool:
        v = os.environ.get(name)
        if v is None or v.strip() == "":
            return default
        return v.strip().lower() in ("1", "true", "yes", "on")

    def _int(name: str, default: int) -> int:
        v = os.environ.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except ValueError:
            return default

    return SchedulerConfig(
        enabled=_bool("SCHEDULER_ENABLED", True),
        ai_interval_seconds=_int("AI_PROCESS_INTERVAL_MIN", 360) * 60,
        monthly_report_hour=_int("MONTHLY_REPORT_HOUR", 2),
        monthly_report_allow_current=_bool("MONTHLY_REPORT_ALLOW_CURRENT", False),
        retry_count=_int("RETRY_COUNT", 3),
        retry_delay_seconds=_int("RETRY_DELAY_SEC", 30),
        lock_timeout_seconds=_int("LOCK_TIMEOUT_SEC", 1800),
        log_level=os.environ.get("SCHEDULER_LOG_LEVEL", "INFO").upper(),
        state_file=os.environ.get("SCHEDULER_STATE_FILE", ".scheduler_state.json"),
    )


# ---------------------------------------------------------------------------
# Job + state store (local JSON; never Firebase)
# ---------------------------------------------------------------------------

@dataclass
class JobState:
    status: str = JOB_IDLE
    last_run: Optional[float] = None
    next_run: Optional[float] = None
    last_successful_ts: Optional[float] = None
    last_error: Optional[str] = None
    locked_at: Optional[float] = None
    locked_by: Optional[int] = None
    last_monthly_report: Optional[str] = None   # "YYYY-MM"


class StateStore:
    """Tiny JSON-backed store for job state + per-job locks."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.data: Dict[str, dict] = {"jobs": {}}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                self.data.setdefault("jobs", {})
            except Exception:  # corrupt -> reset, never crash the scheduler
                logger.warning("Corrupt scheduler state at %s; resetting", self.path)
                self.data = {"jobs": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def job(self, name: str) -> JobState:
        j = self.data["jobs"].get(name)
        if j is None:
            j = asdict(JobState())
            self.data["jobs"][name] = j
        return JobState(**j)

    def set_job(self, name: str, state: JobState) -> None:
        self.data["jobs"][name] = asdict(state)
        self.save()

    def acquire_lock(self, name: str, pid: int, now: float, timeout: int) -> bool:
        j = self.job(name)
        if j.locked_at is not None and j.locked_by != pid and (now - j.locked_at) < timeout:
            return False  # held by another (live) instance
        j.locked_at = now
        j.locked_by = pid
        self.set_job(name, j)
        return True

    def release_lock(self, name: str, pid: int) -> None:
        j = self.job(name)
        if j.locked_by == pid:
            j.locked_at = None
            j.locked_by = None
            self.set_job(name, j)

    def clear_locks(self, pid: int) -> None:
        for name in list(self.data["jobs"].keys()):
            j = self.job(name)
            if j.locked_by == pid:
                j.locked_at = None
                j.locked_by = None
                self.set_job(name, j)


# ---------------------------------------------------------------------------
# Default runners — wrap the EXISTING stage functions only
# ---------------------------------------------------------------------------

def _safe_stage(log, name: str, fn: Callable):
    try:
        log.info("stage started: %s", name)
        result = fn()
        log.info("stage completed: %s", name)
        return result
    except Exception as exc:  # noqa: BLE001 - isolation: one stage must not abort all
        log.error("stage failed: %s (%s)", name, exc)
        return None


def run_ai_pipeline(settings, rate: Optional[float] = None) -> dict:
    """Run Stages 1-2-3-4-5-7-8-9-10-11 in the existing order, reusing the
    canonical stage functions (which persist idempotently). A failure in any
    single stage is isolated: downstream stages that can still run, do."""
    from ai import (preprocessing, anomaly_detection, fault_diagnosis,
                    persist_ai_results, peak_detection, maintenance_risk,
                    forecast, bill_prediction, energy_saving)
    if rate is None:
        rate = float(os.environ.get("BILL_RATE_PER_KWH", "0.0") or "0.0")
    log = logging.getLogger("ai.scheduler.pipeline")

    pre = _safe_stage(log, "preprocessing",
                      lambda: preprocessing.run_preprocessing_pipeline(settings=settings))
    if not pre:
        raise RuntimeError("preprocessing produced no data; aborting AI cycle")

    anomaly_results = _safe_stage(
        log, "anomaly_detection",
        lambda: anomaly_detection.run_anomaly_detection_pipeline(
            settings=settings, preprocess_results=pre))

    _safe_stage(log, "persist_anomalies_faults",
                lambda: persist_ai_results.run_stage_5_pipeline(
                    preprocess_results=pre, anomaly_results=anomaly_results))

    peak_pair = _safe_stage(
        log, "peak_detection",
        lambda: peak_detection.run_peak_detection_pipeline(
            settings=settings, preprocess_results=pre))
    peaks, system_peak = (peak_pair if peak_pair is not None else (None, None))
    _safe_stage(log, "stage7_persist",
                lambda: peak_detection.run_stage_7_pipeline(
                    settings=settings, preprocess_results=pre))

    risk_pair = _safe_stage(
        log, "maintenance_risk",
        lambda: maintenance_risk.run_maintenance_risk_pipeline(
            settings=settings, preprocess_results=pre,
            anomaly_results=anomaly_results, fault_events=None, peak_results=peaks))
    risks, _ = (risk_pair if risk_pair is not None else (None, None))
    _safe_stage(log, "stage8_persist",
                lambda: maintenance_risk.run_stage_8_pipeline(
                    settings=settings, preprocess_results=pre, peak_results=peaks))

    fc_pair = _safe_stage(
        log, "forecast",
        lambda: forecast.run_forecast_pipeline(
            settings=settings, preprocess_results=pre))
    fc_per, fc_system = (fc_pair if fc_pair is not None else (None, None))
    _safe_stage(log, "stage9_persist",
                lambda: forecast.run_stage_9_pipeline(
                    settings=settings, preprocess_results=pre))

    # Stage 10 — bill prediction (depends on the system forecast)
    try:
        actual_kwh = bill_prediction.compute_actual_energy_from_history(settings=settings)
        pred = bill_prediction.predict_bill_from_record(
            actual_kwh, fc_system, horizon="forecast_24h", rate=rate, billing_period="30d")
        if pred.get("status") == "OK" and pred.get("anchor_timestamp"):
            bill_prediction.write_bill_prediction(pred, pred["anchor_timestamp"])
            log.info("stage10: bill prediction persisted")
        else:
            log.warning("stage10: bill prediction withheld (%s)", pred.get("reason"))
    except Exception as exc:
        log.error("stage10: bill prediction failed: %s", exc)

    # Stage 11 — energy-saving suggestions (reuses peaks/risk/forecast)
    meters = {}
    for n in range(1, settings.pzem_count + 1):
        prr = pre.get(n)
        meters[n] = energy_saving.MeterEvidence(
            n,
            feature_frame=(prr.feature_frame if prr and prr.feature_frame is not None else None),
            peak_result=(peaks.get(n) if peaks else None),
            risk_result=(risks.get(n) if risks else None),
            forecast_result=(fc_per.get(n) if fc_per else None),
        )
    try:
        res = energy_saving.run_stage_11_pipeline(meters, rate=rate, force=False)
        log.info("stage11: energy-saving persisted (%d recs)",
                 len(res.get("recommendations", [])))
    except Exception as exc:
        log.error("stage11: energy-saving failed: %s", exc)

    return {"status": "completed"}


def run_monthly_report_job(settings, year: Optional[int] = None,
                           month: Optional[int] = None) -> dict:
    """Build inputs from the live pipelines (idempotent) and emit the monthly
    PDF to local disk only."""
    from ai import report_generator as rg
    data = rg.build_report_input_from_pipelines(settings)
    return rg.generate_monthly_report(data=data, year=year, month=month)


def default_has_new_data(settings, last_ts: Optional[float], now: float) -> bool:
    """Incremental guard: does any PZEM cache hold data newer than the last
    successful run? Cheap local read — no Firebase call."""
    try:
        import pandas as pd
        from ai.data_loader import _cache_path
        max_ts = None
        for n in range(1, settings.pzem_count + 1):
            p = _cache_path(settings, n)
            if p.exists():
                df = pd.read_parquet(p)
                if not df.empty:
                    m = float(df["timestamp"].max())
                    max_ts = m if max_ts is None else max(max_ts, m)
        if max_ts is None:
            return True  # no cache yet -> run once to bootstrap
        return max_ts > (last_ts or 0)
    except Exception:
        return True  # when in doubt, allow a run


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    def __init__(self, config: Optional[SchedulerConfig] = None,
                 settings=None,
                 state_store: Optional[StateStore] = None,
                 clock: Optional[Callable[[], float]] = None,
                 ai_runner: Optional[Callable] = None,
                 monthly_runner: Optional[Callable] = None,
                 has_new_data: Optional[Callable] = None):
        self.config = config or get_scheduler_config()
        self.settings = settings  # resolved lazily if None
        self.state = state_store or StateStore(self.config.state_file)
        self.clock = clock or (lambda: time.time())
        self.ai_runner = ai_runner or run_ai_pipeline
        self.monthly_runner = monthly_runner or run_monthly_report_job
        self.has_new_data = has_new_data or default_has_new_data
        self._pid = os.getpid()

    # -- settings resolution (lazy; needs Firebase creds) -------------------
    def _settings(self):
        if self.settings is None:
            from ai.config import get_settings
            self.settings = get_settings()
        return self.settings

    # -- public tick --------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> None:
        now = now if now is not None else self.clock()
        if not self.config.enabled:
            logger.info("[scheduler] disabled; skipping tick")
            return
        self._tick_ai(now)
        self._tick_monthly(now)

    # -- AI processing job --------------------------------------------------
    def _tick_ai(self, now: float) -> None:
        job = self.state.job("ai_processing")
        if job.next_run is not None and now < job.next_run:
            return  # interval not elapsed -> avoid unnecessary work / writes

        if not self.state.acquire_lock("ai_processing", self._pid, now,
                                       self.config.lock_timeout_seconds):
            logger.info("[ai_processing] skipped: locked by another instance")
            return
        try:
            if (job.last_successful_ts is not None
                    and not self.has_new_data(self._settings(), job.last_successful_ts, now)):
                logger.info("[ai_processing] no new data since last run; skipping execution")
                job.last_run = now
                job.next_run = now + self.config.ai_interval_seconds
                self.state.set_job("ai_processing", job)
                return

            logger.info("[ai_processing] job started")
            self._run_with_retry("ai_processing",
                                 lambda: self.ai_runner(self._settings(), now))
            job.status = JOB_COMPLETED
            job.last_run = now
            job.last_successful_ts = now
            job.last_error = None
            logger.info("[ai_processing] job completed")
        except Exception as exc:  # noqa: BLE001
            job.status = JOB_FAILED
            job.last_error = str(exc)
            job.last_run = now
            logger.error("[ai_processing] job failed: %s", exc)
        finally:
            job.next_run = now + self.config.ai_interval_seconds
            self.state.set_job("ai_processing", job)
            self.state.release_lock("ai_processing", self._pid)

    # -- monthly report job -------------------------------------------------
    @staticmethod
    def monthly_target(now: float, cfg: SchedulerConfig):
        """Pure: which COMPLETED month should be reported, or None.
        On day 1 (UTC) at/after monthly_report_hour we report the previous
        month. Current-month reporting is opt-in (monthly_report_allow_current)."""
        dt = datetime.datetime.fromtimestamp(now, datetime.timezone.utc)
        if dt.day != 1:
            return (dt.year, dt.month) if cfg.monthly_report_allow_current else None
        if dt.hour < cfg.monthly_report_hour:
            return None
        if dt.month == 1:
            return (dt.year - 1, 12)
        return (dt.year, dt.month - 1)

    def _tick_monthly(self, now: float) -> None:
        target = self.monthly_target(now, self.config)
        if target is None:
            return
        key = f"{target[0]}-{target[1]:02d}"
        job = self.state.job("monthly_report")
        if job.last_monthly_report == key:
            return  # already generated for this month -> no duplicate

        if not self.state.acquire_lock("monthly_report", self._pid, now,
                                       self.config.lock_timeout_seconds):
            logger.info("[monthly_report] skipped: locked by another instance")
            return
        try:
            logger.info("[monthly_report] generating for %s", key)
            self._run_with_retry(
                "monthly_report",
                lambda: self.monthly_runner(self._settings(), now, target[0], target[1]))
            job.last_monthly_report = key
            job.status = JOB_COMPLETED
            job.last_error = None
            logger.info("[monthly_report] generated %s", key)
        except Exception as exc:  # noqa: BLE001
            job.status = JOB_FAILED
            job.last_error = str(exc)
            logger.error("[monthly_report] failed: %s", exc)
        finally:
            job.last_run = now
            self.state.set_job("monthly_report", job)
            self.state.release_lock("monthly_report", self._pid)

    # -- retry helper -------------------------------------------------------
    def _run_with_retry(self, name: str, fn: Callable):
        attempts = 0
        last_exc = None
        while attempts <= self.config.retry_count:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                attempts += 1
                if attempts <= self.config.retry_count:
                    logger.warning("[%s] attempt %d/%d failed (%s); retry in %ds",
                                  name, attempts, self.config.retry_count, exc,
                                  self.config.retry_delay_seconds)
                    if self.config.retry_delay_seconds > 0:
                        time.sleep(self.config.retry_delay_seconds)
                else:
                    logger.error("[%s] all %d attempts failed", name, attempts)
        raise last_exc  # type: ignore[misc]

    # -- foreground loop ----------------------------------------------------
    def run_forever(self, poll_seconds: int = 60,
                    stop_event: Optional[threading.Event] = None) -> None:
        logger.info("[scheduler] started pid=%d enabled=%s", self._pid, self.config.enabled)
        if stop_event is None:
            stop_event = threading.Event()

        def _handler(signum, frame):  # noqa: ANN001
            logger.info("[scheduler] shutdown signal received")
            stop_event.set()

        old_int = None
        if threading.current_thread() is threading.main_thread():
            old_int = signal.signal(signal.SIGINT, _handler)
            try:
                signal.signal(signal.SIGTERM, _handler)
            except (ValueError, AttributeError):
                pass

        try:
            while not stop_event.is_set():
                self.tick()
                stop_event.wait(poll_seconds)
        finally:
            self._shutdown()
            if threading.current_thread() is threading.main_thread() and old_int is not None:
                try:
                    signal.signal(signal.SIGINT, old_int)
                except Exception:
                    pass

    def _shutdown(self) -> None:
        logger.info("[scheduler] shutting down; releasing locks")
        self.state.clear_locks(self._pid)
        logger.info("[scheduler] shutdown complete")
