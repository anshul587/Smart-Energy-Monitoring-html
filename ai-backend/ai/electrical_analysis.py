"""
ai/electrical_analysis.py
-------------------------
STAGE 3: Historical Electrical Intelligence.

Reusable historical electrical analysis capability using actual Firebase history:
history/pzem_N/<unix_timestamp>

Reuses existing ai.data_loader and ai.preprocessing infrastructure.
Provides deterministic, validated electrical analysis for:
- Single PZEM historical queries
- Specific date / datetime range / multi-day range
- Multiple PZEM comparison
- System-wide analysis

Calculates: max/min/avg power, peak timestamp, energy consumption,
voltage/current/frequency/PF statistics, hourly/daily analysis,
trends, multi-PZEM comparison, system power/energy analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import data_loader
from . import preprocessing
from .config import Settings, get_settings
from .data_loader import READING_FIELDS, HistoryLoadResult
from .preprocessing import PreprocessResult, preprocess_meter, run_preprocessing_pipeline

logger = logging.getLogger("ai.electrical_analysis")


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Minimum valid rows for meaningful electrical statistics
MIN_ELECTRICAL_ROWS = 2

# Timestamp for "no data" indicator
NO_DATA_TS = -1


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ElectricalStats:
    """Electrical statistics for a single metric (power, voltage, current, etc.)."""
    count: int = 0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    average: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    min_timestamp: Optional[int] = None
    max_timestamp: Optional[int] = None


@dataclass
class EnergyConsumption:
    """Energy consumption calculated from cumulative energy readings."""
    start_energy_kwh: Optional[float] = None
    end_energy_kwh: Optional[float] = None
    consumption_kwh: Optional[float] = None
    start_timestamp: Optional[int] = None
    end_timestamp: Optional[int] = None
    valid: bool = False


@dataclass
class HourlyAnalysis:
    """Hourly electrical analysis for a PZEM."""
    hour: int  # 0-23 (UTC)
    power: ElectricalStats
    voltage: ElectricalStats
    current: ElectricalStats
    frequency: ElectricalStats
    pf: ElectricalStats
    sample_count: int = 0


@dataclass
class DailyAnalysis:
    """Daily electrical analysis for a PZEM."""
    date: str  # YYYY-MM-DD (UTC)
    power: ElectricalStats
    voltage: ElectricalStats
    current: ElectricalStats
    frequency: ElectricalStats
    pf: ElectricalStats
    energy_consumption: EnergyConsumption
    sample_count: int = 0


@dataclass
class TrendAnalysis:
    """Trend analysis over the historical window."""
    power_trend_per_hour: Optional[float] = None  # W/h
    current_trend_per_hour: Optional[float] = None  # A/h
    voltage_trend_per_hour: Optional[float] = None  # V/h
    pf_trend_per_hour: Optional[float] = None  # PF/h
    frequency_trend_per_hour: Optional[float] = None  # Hz/h
    data_span_hours: float = 0.0
    sample_count: int = 0


@dataclass
class PZEMHistoricalResult:
    """Complete historical analysis for one PZEM."""
    pzem_number: int
    status: str  # "OK", "NO_DATA", "INSUFFICIENT_DATA", "ERROR"
    reason: Optional[str] = None

    # Time window
    requested_start: Optional[int] = None
    requested_end: Optional[int] = None
    actual_start: Optional[int] = None
    actual_end: Optional[int] = None
    available_days: float = 0.0

    # Electrical statistics
    power: ElectricalStats = field(default_factory=ElectricalStats)
    voltage: ElectricalStats = field(default_factory=ElectricalStats)
    current: ElectricalStats = field(default_factory=ElectricalStats)
    frequency: ElectricalStats = field(default_factory=ElectricalStats)
    pf: ElectricalStats = field(default_factory=ElectricalStats)

    # Energy consumption
    energy_consumption: EnergyConsumption = field(default_factory=EnergyConsumption)

    # Hourly / Daily analysis
    hourly: list[HourlyAnalysis] = field(default_factory=list)
    daily: list[DailyAnalysis] = field(default_factory=list)

    # Trends
    trend: TrendAnalysis = field(default_factory=TrendAnalysis)

    # Raw data info
    sample_count: int = 0
    valid_rows: int = 0
    dropped_rows: int = 0

    # Filtered data frame for system aggregation (300-s bucket alignment).
    # Stored by analyze_pzem_history(); used by _aggregate_to_system().
    frame: Optional[pd.DataFrame] = None


@dataclass
class SystemHistoricalResult:
    """System-wide historical analysis (aggregate of all PZEMs)."""
    status: str
    reason: Optional[str] = None

    requested_start: Optional[int] = None
    requested_end: Optional[int] = None

    total_power: ElectricalStats = field(default_factory=ElectricalStats)
    total_energy_kwh: Optional[float] = None

    per_pzem: dict[int, PZEMHistoricalResult] = field(default_factory=dict)
    meters_analyzed: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_pzem_number(pzem_number: int, settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    if not (1 <= pzem_number <= settings.pzem_count):
        raise ValueError(
            f"pzem_number must be between 1 and {settings.pzem_count}, got {pzem_number}"
        )


def _parse_date_to_timestamp(date_str: str) -> int:
    """Parse YYYY-MM-DD date string to UTC midnight timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _parse_datetime_to_timestamp(dt_str: str) -> int:
    """Parse ISO datetime string (YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS) to timestamp."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    raise ValueError(f"Invalid datetime format: {dt_str}. Use YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS")


def _filter_by_timerange(frame: pd.DataFrame, start_ts: Optional[int], end_ts: Optional[int]) -> pd.DataFrame:
    """Filter DataFrame by timestamp range (inclusive)."""
    if frame.empty:
        return frame
    mask = pd.Series(True, index=frame.index)
    if start_ts is not None:
        mask &= frame["timestamp"] >= start_ts
    if end_ts is not None:
        mask &= frame["timestamp"] <= end_ts
    return frame[mask].copy()


def _enrich_with_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_day and date_utc columns for hourly/daily analysis."""
    if frame.empty:
        return frame
    frame = frame.copy()
    dt_utc = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    frame["hour_of_day"] = dt_utc.dt.hour
    frame["date_utc"] = dt_utc.dt.date.astype(str)
    return frame


def _calculate_electrical_stats(series: pd.Series, timestamps: pd.Series) -> ElectricalStats:
    """Calculate statistics for a single electrical metric."""
    if series.empty:
        return ElectricalStats()

    clean = series.dropna()
    if clean.empty:
        return ElectricalStats()

    ts_clean = timestamps.loc[clean.index]

    min_idx = clean.idxmin()
    max_idx = clean.idxmax()

    return ElectricalStats(
        count=len(clean),
        minimum=float(clean.min()),
        maximum=float(clean.max()),
        average=float(clean.mean()),
        median=float(clean.median()),
        std_dev=float(clean.std()) if len(clean) > 1 else 0.0,
        min_timestamp=int(ts_clean.loc[min_idx]) if min_idx is not None else None,
        max_timestamp=int(ts_clean.loc[max_idx]) if max_idx is not None else None,
    )


def _calculate_energy_consumption(energy_series: pd.Series, timestamps: pd.Series) -> EnergyConsumption:
    """Calculate energy consumption from cumulative energy readings."""
    if energy_series.empty:
        return EnergyConsumption()

    clean_energy = energy_series.dropna()
    if clean_energy.empty or len(clean_energy) < 2:
        return EnergyConsumption()

    # Ensure sorted by timestamp
    ts_clean = timestamps.loc[clean_energy.index]
    combined = pd.DataFrame({"timestamp": ts_clean, "energy": clean_energy}).sort_values("timestamp")

    start_energy = float(combined["energy"].iloc[0])
    end_energy = float(combined["energy"].iloc[-1])
    start_ts = int(combined["timestamp"].iloc[0])
    end_ts = int(combined["timestamp"].iloc[-1])

    consumption = end_energy - start_energy
    if consumption < 0:
        # Counter reset - clamp to 0 and flag
        logger.warning(f"Energy counter decreased from {start_energy} to {end_energy} kWh; clamping to 0")
        consumption = 0.0

    return EnergyConsumption(
        start_energy_kwh=start_energy,
        end_energy_kwh=end_energy,
        consumption_kwh=consumption,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        valid=True,
    )


def _compute_hourly_analysis(frame: pd.DataFrame) -> list[HourlyAnalysis]:
    """Compute hourly electrical analysis."""
    if frame.empty:
        return []

    frame = _enrich_with_time_columns(frame)
    hours = frame.groupby("hour_of_day")
    results = []

    for hour, group in hours:
        if group.empty:
            continue

        power_stats = _calculate_electrical_stats(group["power"], group["timestamp"])
        voltage_stats = _calculate_electrical_stats(group["voltage"], group["timestamp"])
        current_stats = _calculate_electrical_stats(group["current"], group["timestamp"])
        freq_stats = _calculate_electrical_stats(group["frequency"], group["timestamp"])
        pf_stats = _calculate_electrical_stats(group["pf"], group["timestamp"])

        results.append(HourlyAnalysis(
            hour=int(hour),
            power=power_stats,
            voltage=voltage_stats,
            current=current_stats,
            frequency=freq_stats,
            pf=pf_stats,
            sample_count=len(group),
        ))

    return sorted(results, key=lambda h: h.hour)


def _compute_daily_analysis(frame: pd.DataFrame) -> list[DailyAnalysis]:
    """Compute daily electrical analysis with energy consumption."""
    if frame.empty:
        return []

    frame = _enrich_with_time_columns(frame)
    days = frame.groupby("date_utc")
    results = []

    for date_str, group in days:
        if group.empty:
            continue

        power_stats = _calculate_electrical_stats(group["power"], group["timestamp"])
        voltage_stats = _calculate_electrical_stats(group["voltage"], group["timestamp"])
        current_stats = _calculate_electrical_stats(group["current"], group["timestamp"])
        freq_stats = _calculate_electrical_stats(group["frequency"], group["timestamp"])
        pf_stats = _calculate_electrical_stats(group["pf"], group["timestamp"])
        energy_cons = _calculate_energy_consumption(group["energy"], group["timestamp"])

        results.append(DailyAnalysis(
            date=date_str,
            power=power_stats,
            voltage=voltage_stats,
            current=current_stats,
            frequency=freq_stats,
            pf=pf_stats,
            energy_consumption=energy_cons,
            sample_count=len(group),
        ))

    return sorted(results, key=lambda d: d.date)


def _compute_trend(frame: pd.DataFrame) -> TrendAnalysis:
    """Compute linear trends over the analysis window."""
    if frame.empty or len(frame) < MIN_ELECTRICAL_ROWS:
        return TrendAnalysis()

    elapsed_hours = (frame["timestamp"] - frame["timestamp"].iloc[0]) / 3600.0
    span_hours = float(elapsed_hours.iloc[-1])

    trend = TrendAnalysis(data_span_hours=span_hours, sample_count=len(frame))

    for col, attr in [
        ("power", "power_trend_per_hour"),
        ("current", "current_trend_per_hour"),
        ("voltage", "voltage_trend_per_hour"),
        ("pf", "pf_trend_per_hour"),
        ("frequency", "frequency_trend_per_hour"),
    ]:
        if col in frame.columns:
            y = frame[col].dropna()
            if len(y) >= MIN_ELECTRICAL_ROWS:
                x = elapsed_hours.loc[y.index]
                if x.var() > 0:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        slope = x.cov(y) / x.var()
                    if np.isfinite(slope):
                        setattr(trend, attr, float(slope))

    return trend


def _aggregate_to_system(pzem_results: dict[int, PZEMHistoricalResult]) -> SystemHistoricalResult:
    """Aggregate per-PZEM results into system-wide analysis.

    Uses 300-second (5-minute) timestamp bucket alignment so that system
    power is a true simultaneous sum: only buckets where every included
    PZEM has at least one valid sample are kept, and the system power at
    each such bucket is the sum of all PZEMs' power in that bucket.
    """
    HISTORY_SLOT_SECONDS = 300

    valid_results = {
        n: r for n, r in pzem_results.items()
        if r.status == "OK"
        and r.sample_count > 0
        and r.frame is not None
        and not r.frame.empty
    }

    if not valid_results:
        return SystemHistoricalResult(status="NO_DATA", reason="No PZEMs with valid data")

    # Build a per-PZEM slot-and-power table from each PZEM's filtered frame.
    # Only keep rows with valid (non-NaN, non-negative) power.
    pzem_slot_power: dict[int, pd.DataFrame] = {}
    for n, r in valid_results.items():
        frame = r.frame
        work = frame[frame["power"].notna() & (frame["power"] >= 0)].copy()
        if work.empty:
            continue
        work = work[["timestamp", "power"]].copy()
        work["slot"] = (work["timestamp"] // HISTORY_SLOT_SECONDS) * HISTORY_SLOT_SECONDS
        # One row per slot per meter: take the first power value in the slot
        work = work.groupby("slot", as_index=False).first()
        pzem_slot_power[n] = work

    if not pzem_slot_power:
        return SystemHistoricalResult(status="NO_DATA", reason="No PZEMs with valid power data")

    # Inner-join all PZEM slot tables on slot so that only buckets where
    # EVERY included PZEM has at least one sample are retained.
    merged: Optional[pd.DataFrame] = None
    for n in sorted(pzem_slot_power):
        part = pzem_slot_power[n].rename(columns={"power": f"p{n}_power"})[["slot", f"p{n}_power"]]
        part = part.groupby("slot", as_index=False).max()
        merged = part if merged is None else merged.merge(part, on="slot", how="inner")

    if merged is None or merged.empty:
        return SystemHistoricalResult(
            status="NO_DATA",
            reason="No common 300-second slot had valid samples from every included PZEM.",
        )

    # Sum power across all included PZEMs in each common slot.
    pcol_names = [f"p{n}_power" for n in sorted(pzem_slot_power)]
    merged["total_power"] = merged[pcol_names].sum(axis=1)

    # System statistics from the aligned simultaneous power series.
    total_power_values = merged["total_power"].tolist()
    slot_values = merged["slot"].tolist()

    system_power = ElectricalStats(
        count=len(total_power_values),
        minimum=float(min(total_power_values)),
        maximum=float(max(total_power_values)),
        average=float(np.mean(total_power_values)),
        median=float(np.median(total_power_values)) if total_power_values else None,
        std_dev=float(np.std(total_power_values)) if len(total_power_values) > 1 else 0.0,
        min_timestamp=int(slot_values[total_power_values.index(min(total_power_values))]),
        max_timestamp=int(slot_values[total_power_values.index(max(total_power_values))]),
    )

    # System energy is additive: total system energy = sum of each PZEM's
    # interval consumption_kWh.  Energy is cumulative per PZEM and additive
    # because it represents total kWh consumed per meter over the window,
    # unlike instantaneous power which requires timestamp alignment.
    total_energy = sum(
        r.energy_consumption.consumption_kwh or 0
        for r in valid_results.values()
        if r.energy_consumption.valid
    )

    requested_start = min(
        (r.requested_start for r in valid_results.values() if r.requested_start is not None),
        default=None,
    )
    requested_end = max(
        (r.requested_end for r in valid_results.values() if r.requested_end is not None),
        default=None,
    )

    return SystemHistoricalResult(
        status="OK",
        requested_start=requested_start,
        requested_end=requested_end,
        total_power=system_power,
        total_energy_kwh=total_energy if total_energy > 0 else None,
        per_pzem={n: r for n, r in valid_results.items()},
        meters_analyzed=len(valid_results),
    )


def _build_error_result(pzem_number: int, reason: str, start_ts: Optional[int], end_ts: Optional[int]) -> PZEMHistoricalResult:
    """Build an error result for a PZEM."""
    return PZEMHistoricalResult(
        pzem_number=pzem_number,
        status="ERROR",
        reason=reason,
        requested_start=start_ts,
        requested_end=end_ts,
    )


def _build_no_data_result(pzem_number: int, start_ts: Optional[int], end_ts: Optional[int]) -> PZEMHistoricalResult:
    """Build a no-data result for a PZEM."""
    return PZEMHistoricalResult(
        pzem_number=pzem_number,
        status="NO_DATA",
        reason="No historical data available for the requested period",
        requested_start=start_ts,
        requested_end=end_ts,
    )


def _build_insufficient_result(pzem_number: int, reason: str, start_ts: Optional[int], end_ts: Optional[int],
                                sample_count: int) -> PZEMHistoricalResult:
    """Build an insufficient data result for a PZEM."""
    return PZEMHistoricalResult(
        pzem_number=pzem_number,
        status="INSUFFICIENT_DATA",
        reason=reason,
        requested_start=start_ts,
        requested_end=end_ts,
        sample_count=sample_count,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_pzem_history(
    pzem_number: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    settings: Optional[Settings] = None,
    history_result: Optional[HistoryLoadResult] = None,
) -> PZEMHistoricalResult:
    """
    Analyze historical electrical data for a single PZEM.

    Args:
        pzem_number: PZEM meter number (1-9)
        start: Start timestamp (unix seconds, inclusive). None = use all available.
        end: End timestamp (unix seconds, inclusive). None = use all available.
        settings: Optional Settings override.
        history_result: Optional pre-loaded HistoryLoadResult (for testing).

    Returns:
        PZEMHistoricalResult with complete electrical analysis.
    """
    settings = settings or get_settings()
    _validate_pzem_number(pzem_number, settings)

    # Load history
    if history_result is None:
        history_result = data_loader.fetch_meter_history(pzem_number, settings=settings)

    frame = history_result.frame
    if frame.empty:
        return _build_no_data_result(pzem_number, start, end)

    # Filter by requested time range
    filtered = _filter_by_timerange(frame, start, end)

    if filtered.empty:
        return _build_no_data_result(pzem_number, start, end)

    if len(filtered) < MIN_ELECTRICAL_ROWS:
        return _build_insufficient_result(
            pzem_number,
            f"Only {len(filtered)} sample(s) in range; need >= {MIN_ELECTRICAL_ROWS}",
            start, end, len(filtered)
        )

    actual_start = int(filtered["timestamp"].iloc[0])
    actual_end = int(filtered["timestamp"].iloc[-1])
    available_days = round((actual_end - actual_start) / 86400, 2)

    # Electrical statistics
    power_stats = _calculate_electrical_stats(filtered["power"], filtered["timestamp"])
    voltage_stats = _calculate_electrical_stats(filtered["voltage"], filtered["timestamp"])
    current_stats = _calculate_electrical_stats(filtered["current"], filtered["timestamp"])
    freq_stats = _calculate_electrical_stats(filtered["frequency"], filtered["timestamp"])
    pf_stats = _calculate_electrical_stats(filtered["pf"], filtered["timestamp"])

    # Energy consumption
    energy_cons = _calculate_energy_consumption(filtered["energy"], filtered["timestamp"])

    # Hourly / Daily analysis
    hourly = _compute_hourly_analysis(filtered)
    daily = _compute_daily_analysis(filtered)

    # Trends
    trend = _compute_trend(filtered)

    result = PZEMHistoricalResult(
        pzem_number=pzem_number,
        status="OK",
        requested_start=start,
        requested_end=end,
        actual_start=actual_start,
        actual_end=actual_end,
        available_days=available_days,
        power=power_stats,
        voltage=voltage_stats,
        current=current_stats,
        frequency=freq_stats,
        pf=pf_stats,
        energy_consumption=energy_cons,
        hourly=hourly,
        daily=daily,
        trend=trend,
        sample_count=len(filtered),
        valid_rows=len(filtered),
        dropped_rows=history_result.dropped_rows,
        frame=filtered,
    )
    return result


def analyze_pzem_by_date(
    pzem_number: int,
    date: str,  # YYYY-MM-DD
    settings: Optional[Settings] = None,
) -> PZEMHistoricalResult:
    """Analyze a single PZEM for a specific date (UTC)."""
    start_ts = _parse_date_to_timestamp(date)
    end_ts = start_ts + 86399  # End of day
    return analyze_pzem_history(pzem_number, start=start_ts, end=end_ts, settings=settings)


def analyze_pzem_by_daterange(
    pzem_number: int,
    start_date: str,  # YYYY-MM-DD
    end_date: str,    # YYYY-MM-DD (inclusive)
    settings: Optional[Settings] = None,
) -> PZEMHistoricalResult:
    """Analyze a single PZEM for a date range (UTC)."""
    start_ts = _parse_date_to_timestamp(start_date)
    end_ts = _parse_date_to_timestamp(end_date) + 86399
    return analyze_pzem_history(pzem_number, start=start_ts, end=end_ts, settings=settings)


def analyze_pzem_by_datetime_range(
    pzem_number: int,
    start_datetime: str,  # YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS
    end_datetime: str,
    settings: Optional[Settings] = None,
) -> PZEMHistoricalResult:
    """Analyze a single PZEM for a precise datetime range (UTC)."""
    start_ts = _parse_datetime_to_timestamp(start_datetime)
    end_ts = _parse_datetime_to_timestamp(end_datetime)
    if start_ts > end_ts:
        raise ValueError("start_datetime must be <= end_datetime")
    return analyze_pzem_history(pzem_number, start=start_ts, end=end_ts, settings=settings)


def analyze_multiple_pzems(
    pzem_numbers: list[int],
    start: Optional[int] = None,
    end: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> dict[int, PZEMHistoricalResult]:
    """Analyze multiple PZEMs for comparison."""
    settings = settings or get_settings()
    results = {}
    for n in pzem_numbers:
        if not (1 <= n <= settings.pzem_count):
            results[n] = _build_error_result(n, f"Invalid PZEM number (1-{settings.pzem_count})", start, end)
            continue
        results[n] = analyze_pzem_history(n, start=start, end=end, settings=settings)
    return results


def analyze_system_history(
    start: Optional[int] = None,
    end: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> SystemHistoricalResult:
    """Analyze system-wide historical data (all configured PZEMs)."""
    settings = settings or get_settings()

    pzem_results: dict[int, PZEMHistoricalResult] = {}
    for n in range(1, settings.pzem_count + 1):
        pzem_results[n] = analyze_pzem_history(n, start=start, end=end, settings=settings)

    system_result = _aggregate_to_system(pzem_results)
    system_result.requested_start = start
    system_result.requested_end = end
    return system_result


# ---------------------------------------------------------------------------
# Reporting / Serialization
# ---------------------------------------------------------------------------


def _stats_to_dict(stats: ElectricalStats) -> dict:
    return {
        "count": stats.count,
        "minimum": stats.minimum,
        "maximum": stats.maximum,
        "average": stats.average,
        "median": stats.median,
        "std_dev": stats.std_dev,
        "min_timestamp": stats.min_timestamp,
        "max_timestamp": stats.max_timestamp,
    }


def _energy_to_dict(ec: EnergyConsumption) -> dict:
    return {
        "start_energy_kwh": ec.start_energy_kwh,
        "end_energy_kwh": ec.end_energy_kwh,
        "consumption_kwh": ec.consumption_kwh,
        "start_timestamp": ec.start_timestamp,
        "end_timestamp": ec.end_timestamp,
        "valid": ec.valid,
    }


def _hourly_to_dict(h: HourlyAnalysis) -> dict:
    return {
        "hour": h.hour,
        "power": _stats_to_dict(h.power),
        "voltage": _stats_to_dict(h.voltage),
        "current": _stats_to_dict(h.current),
        "frequency": _stats_to_dict(h.frequency),
        "pf": _stats_to_dict(h.pf),
        "sample_count": h.sample_count,
    }


def _daily_to_dict(d: DailyAnalysis) -> dict:
    return {
        "date": d.date,
        "power": _stats_to_dict(d.power),
        "voltage": _stats_to_dict(d.voltage),
        "current": _stats_to_dict(d.current),
        "frequency": _stats_to_dict(d.frequency),
        "pf": _stats_to_dict(d.pf),
        "energy_consumption": _energy_to_dict(d.energy_consumption),
        "sample_count": d.sample_count,
    }


def _trend_to_dict(t: TrendAnalysis) -> dict:
    return {
        "power_trend_per_hour": t.power_trend_per_hour,
        "current_trend_per_hour": t.current_trend_per_hour,
        "voltage_trend_per_hour": t.voltage_trend_per_hour,
        "pf_trend_per_hour": t.pf_trend_per_hour,
        "frequency_trend_per_hour": t.frequency_trend_per_hour,
        "data_span_hours": t.data_span_hours,
        "sample_count": t.sample_count,
    }


def pzem_result_to_dict(r: PZEMHistoricalResult) -> dict:
    """Serialize PZEMHistoricalResult to JSON-safe dict."""
    return {
        "pzem_number": r.pzem_number,
        "status": r.status,
        "reason": r.reason,
        "requested_start": r.requested_start,
        "requested_end": r.requested_end,
        "actual_start": r.actual_start,
        "actual_end": r.actual_end,
        "available_days": r.available_days,
        "power": _stats_to_dict(r.power),
        "voltage": _stats_to_dict(r.voltage),
        "current": _stats_to_dict(r.current),
        "frequency": _stats_to_dict(r.frequency),
        "pf": _stats_to_dict(r.pf),
        "energy_consumption": _energy_to_dict(r.energy_consumption),
        "hourly": [_hourly_to_dict(h) for h in r.hourly],
        "daily": [_daily_to_dict(d) for d in r.daily],
        "trend": _trend_to_dict(r.trend),
        "sample_count": r.sample_count,
        "valid_rows": r.valid_rows,
        "dropped_rows": r.dropped_rows,
    }


def system_result_to_dict(r: SystemHistoricalResult) -> dict:
    """Serialize SystemHistoricalResult to JSON-safe dict."""
    return {
        "status": r.status,
        "reason": r.reason,
        "requested_start": r.requested_start,
        "requested_end": r.requested_end,
        "total_power": _stats_to_dict(r.total_power),
        "total_energy_kwh": r.total_energy_kwh,
        "per_pzem": {str(k): pzem_result_to_dict(v) for k, v in r.per_pzem.items()},
        "meters_analyzed": r.meters_analyzed,
    }


def format_pzem_report(r: PZEMHistoricalResult) -> str:
    """Human-readable report for one PZEM."""
    lines = [f"PZEM {r.pzem_number} Historical Analysis"]
    lines.append(f"Status: {r.status}")
    if r.reason:
        lines.append(f"Reason: {r.reason}")
    lines.append(f"Requested window: {r.requested_start} to {r.requested_end}")
    lines.append(f"Actual data: {r.actual_start} to {r.actual_end} ({r.available_days:.2f} days)")
    lines.append(f"Samples: {r.sample_count} (valid: {r.valid_rows}, dropped: {r.dropped_rows})")
    lines.append("")

    for label, stats in [("Power (W)", r.power), ("Voltage (V)", r.voltage),
                          ("Current (A)", r.current), ("Frequency (Hz)", r.frequency),
                          ("Power Factor", r.pf)]:
        if stats.count > 0:
            lines.append(f"  {label}: min={stats.minimum:.2f} @ {stats.min_timestamp}, "
                         f"max={stats.maximum:.2f} @ {stats.max_timestamp}, "
                         f"avg={stats.average:.2f}, median={stats.median:.2f}, "
                         f"std={stats.std_dev:.2f} (n={stats.count})")

    if r.energy_consumption.valid:
        lines.append(f"  Energy: {r.energy_consumption.consumption_kwh:.4f} kWh consumed "
                     f"({r.energy_consumption.start_energy_kwh:.4f} -> {r.energy_consumption.end_energy_kwh:.4f} kWh)")

    if r.hourly:
        lines.append(f"\n  Hourly breakdown ({len(r.hourly)} hours with data):")
        for h in r.hourly:
            if h.power.count > 0:
                lines.append(f"    Hour {h.hour:02d}: power avg={h.power.average:.1f}W max={h.power.maximum:.1f}W (n={h.sample_count})")

    if r.daily:
        lines.append(f"\n  Daily breakdown ({len(r.daily)} days with data):")
        for d in r.daily:
            if d.power.count > 0:
                lines.append(f"    {d.date}: power avg={d.power.average:.1f}W max={d.power.maximum:.1f}W "
                             f"energy={d.energy_consumption.consumption_kwh:.4f}kWh (n={d.sample_count})")

    if r.trend.sample_count > 0:
        lines.append(f"\n  Trends (over {r.trend.data_span_hours:.1f}h): "
                     f"power={r.trend.power_trend_per_hour:.3f} W/h, "
                     f"current={r.trend.current_trend_per_hour:.4f} A/h, "
                     f"voltage={r.trend.voltage_trend_per_hour:.3f} V/h, "
                     f"pf={r.trend.pf_trend_per_hour:.5f}/h")

    return "\n".join(lines)


def format_system_report(r: SystemHistoricalResult) -> str:
    """Human-readable system-wide report."""
    lines = ["SYSTEM-WIDE Historical Analysis"]
    lines.append(f"Status: {r.status}")
    if r.reason:
        lines.append(f"Reason: {r.reason}")
    lines.append(f"Meters analyzed: {r.meters_analyzed}/{len(r.per_pzem)}")
    lines.append("")

    if r.total_power.count > 0:
        lines.append(f"Total Power: min={r.total_power.minimum:.1f}W, "
                     f"max={r.total_power.maximum:.1f}W, avg={r.total_power.average:.1f}W")
    if r.total_energy_kwh is not None:
        lines.append(f"Total Energy Consumption: {r.total_energy_kwh:.4f} kWh")

    lines.append("\nPer-PZEM Summary:")
    for n in sorted(r.per_pzem):
        p = r.per_pzem[n]
        if p.status == "OK":
            lines.append(f"  PZEM {n}: power avg={p.power.average:.1f}W max={p.power.maximum:.1f}W "
                         f"energy={p.energy_consumption.consumption_kwh:.4f}kWh (n={p.sample_count})")
        else:
            lines.append(f"  PZEM {n}: {p.status} - {p.reason}")

    return "\n".join(lines)