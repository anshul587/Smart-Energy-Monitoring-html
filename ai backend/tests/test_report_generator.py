"""
tests/test_report_generator.py — Stage 13 MONTHLY report generator (PDF only).
Deterministic, no Firebase / no network. Synthetic frames + demo input only.
"""
from __future__ import annotations

import datetime
import os

import numpy as np
import pandas as pd
import pytest

from ai import report_generator as rg
from ai.report_generator import ReportInput, build_report, _safe, _fmt

UTC = datetime.timezone.utc
DASHBOARD = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "..", "Dashboard Smart-Monitoring-System", "index.html"))


def _frame(n=288, power=200.0, slot=300, base_ts=None):
    if base_ts is None:
        base_ts = int(datetime.datetime(2026, 8, 1, tzinfo=UTC).timestamp())
    ts = np.arange(base_ts, base_ts + n * slot, slot)
    return pd.DataFrame({
        "timestamp": ts,
        "voltage": 230.0,
        "current": power / 230.0,
        "power": np.full(n, power, dtype=float),
        "energy": np.cumsum(np.full(n, power)) * slot / 3600.0,
        "frequency": 50.0,
        "pf": 0.95,
    })


def _month_period(year=2026, month=8):
    s, e = rg._month_bounds(year, month)
    return s, e


# --- safe numeric helpers --------------------------------------------------

def test_safe_nan_is_none():
    assert _safe(float("nan")) is None

def test_safe_inf_is_none():
    assert _safe(float("inf")) is None

def test_safe_good_rounds():
    assert _safe(1.23456, 2) == 1.23

def test_fmt_none_default():
    assert _fmt(None) == "—"

def test_fmt_value():
    assert _fmt(1.234, 2) == "1.23"


# --- period extraction -----------------------------------------------------

def test_pzem_stats_energy_and_peak():
    f = _frame(n=4, power=100.0, slot=300)
    start, end = _month_period()
    st = rg._pzem_stats(f, start, end)
    assert st is not None
    # 4 samples * 100W * 300s = 120000 Ws /1000 /3600 = 0.0333 kWh
    assert abs(st["energy_kwh"] - 0.0333) < 1e-3
    assert st["peak_power_w"] == 100.0
    assert st["samples"] == 4

def test_pzem_stats_outside_period_none():
    f = _frame(n=10, power=100.0)
    start, end = rg._month_bounds(2027, 1)
    assert rg._pzem_stats(f, start, end) is None

def test_count_in_period_filters():
    a = [{"pzem_number": 1, "timestamp": 100}, {"pzem_number": 1, "timestamp": 9_999_999_999}]
    c, listed = rg._count_in_period(a, 0, 1000)
    assert c == 1 and len(listed) == 1

def test_peak_in_period_included():
    from types import SimpleNamespace
    pk = SimpleNamespace(status="PEAK_FOUND", peak_power_w=500.0,
                        peak_timestamp=100, baseline_power_w=100.0, peak_above_baseline_w=400.0)
    assert rg._peak_in_period(pk, 0, 1000) is not None
    assert rg._peak_in_period(pk, 5000, 6000) is None

def test_peak_dict_status_no_peak():
    assert rg._peak_in_period({"status": "NO_PEAK"}, 0, 1000) is None

def test_forecast_summary_from_list():
    fc = {"forecast_24h": {"status": "FORECAST", "forecast_power_w": [100, 200, 300]}}
    out = rg._forecast_summary(fc)
    assert out["status"] == "FORECAST"
    # (100+200+300)=600W*300s/1000/3600 = 0.05 kWh
    assert abs(out["forecast_energy_kwh"] - 0.05) < 1e-6

def test_bill_summary_none():
    assert rg._bill_summary(None)["status"] == "NO_BILL"


# --- monthly report building ----------------------------------------------

def test_build_report_monthly_system_present():
    data = ReportInput(pzem_count=3, frames={1: _frame(power=200.0)})
    start, end = _month_period()
    rep = build_report(data, start, end, "monthly")
    assert rep["system"] is not None
    assert rep["active_pzem"] == 1
    assert rep["total_pzem"] == 3
    assert rep["kind"] == "monthly"

def test_build_report_no_frames():
    rep = build_report(ReportInput(pzem_count=2), *_month_period(), "monthly")
    assert rep["system"] is None

def test_build_report_alerts_emergency_first():
    faults = {1: [
        {"pzem_number": 1, "timestamp": 500, "fault_type": "X", "severity": "WARNING"},
        {"pzem_number": 1, "timestamp": 100, "fault_type": "Y", "severity": "EMERGENCY"},
    ]}
    rep = build_report(ReportInput(pzem_count=1, faults=faults), 0, 1000, "monthly")
    assert rep["alerts"][0]["severity"] == "EMERGENCY"

def test_monthly_daily_energy_trend_present():
    f = _frame(n=288 * 2, power=200.0)  # 2 days
    data = ReportInput(pzem_count=1, frames={1: f})
    start, end = rg._month_bounds(2026, 8)
    rep = build_report(data, start, end, "monthly")
    assert "daily_energy_trend" in rep["charts"]
    assert "peak_trend" in rep["charts"]

def test_daily_power_trend_absent_in_monthly():
    f = _frame(n=288, power=200.0)
    data = ReportInput(pzem_count=1, frames={1: f})
    start, end = _month_period()
    rep = build_report(data, start, end, "monthly")
    assert "power_trend" not in rep["charts"]

def test_event_summary_chart_when_events():
    faults = {1: [{"pzem_number": 1, "timestamp": 500, "fault_type": "X", "severity": "WARNING"}]}
    anomalies = {1: [{"pzem_number": 1, "timestamp": 600, "type": "a", "severity": "ANOMALY"}]}
    rep = build_report(ReportInput(pzem_count=1, faults=faults, anomalies=anomalies), 0, 1000, "monthly")
    assert "event_summary" in rep["charts"]


# --- monthly PDF generation -------------------------------------------------

def test_generate_monthly_creates_pdf(tmp_path):
    data = ReportInput(pzem_count=1, frames={1: _frame(n=288 * 3, power=200.0)})
    res = rg.generate_monthly_report(data=data, year=2026, month=8, output_dir=str(tmp_path))
    assert os.path.exists(res["pdf"])
    assert res["stub"] == "report-2026-08"
    # no html output
    assert "html" not in res

def test_monthly_pdf_valid_header(tmp_path):
    data = ReportInput(pzem_count=1, frames={1: _frame(power=200.0)})
    res = rg.generate_monthly_report(data=data, year=2026, month=8, output_dir=str(tmp_path))
    with open(res["pdf"], "rb") as fh:
        assert fh.read(8) == b"%PDF-1.4"

def test_monthly_latest_pdf_created(tmp_path):
    data = ReportInput(pzem_count=1, frames={1: _frame(power=200.0)})
    res = rg.generate_monthly_report(data=data, year=2026, month=8, output_dir=str(tmp_path))
    out_dir = os.path.join(str(tmp_path), "monthly")
    assert os.path.exists(os.path.join(out_dir, "latest.pdf"))
    assert not os.path.exists(os.path.join(out_dir, "latest.html"))

def test_monthly_pdf_contains_charts_and_text(tmp_path):
    data = ReportInput(pzem_count=1, frames={1: _frame(n=288 * 3, power=200.0)})
    res = rg.generate_monthly_report(data=data, year=2026, month=8, output_dir=str(tmp_path))
    # PDF should embed vector chart operators when charts exist
    with open(res["pdf"], "rb") as fh:
        blob = fh.read()
    assert b"Monthly Energy Report" in blob
    # chart path ops (rectangle fill 're f' or line stroke 'S') present
    assert b"re f" in blob or b" l S" in blob

def test_monthly_contents_present(tmp_path):
    data = ReportInput(pzem_count=2, frames={1: _frame(power=200.0)})
    res = rg.generate_monthly_report(data=data, year=2026, month=8, output_dir=str(tmp_path))
    rep = res["report"]
    assert rep["system"] is not None
    assert len(rep["pzem_rows"]) == 2
    assert "recommendations" in rep["ai_insights"]

def test_missing_data_no_crash(tmp_path):
    res = rg.generate_monthly_report(data=ReportInput(pzem_count=1),
                                     year=2026, month=8, output_dir=str(tmp_path))
    assert os.path.getsize(res["pdf"]) > 0
    assert res["report"]["system"] is None


# --- demo & determinism ----------------------------------------------------

def test_demo_input_reproducible():
    a = rg.demo_input(seed=7)
    b = rg.demo_input(seed=7)
    s, e = rg._month_bounds(2026, 8)
    sa = rg._system_power(a.frames, s, e)
    sb = rg._system_power(b.frames, s, e)
    assert sa is not None and sb is not None
    assert np.allclose(sa.dropna().values, sb.dropna().values)

def test_demo_input_has_recommendations():
    d = rg.demo_input()
    assert len(d.recommendations) > 0

def test_monthly_deterministic_bytes(tmp_path):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    r1 = rg.generate_monthly_report(data=rg.demo_input(), year=2026, month=8, output_dir=str(d1))
    r2 = rg.generate_monthly_report(data=rg.demo_input(), year=2026, month=8, output_dir=str(d2))
    with open(r1["pdf"], "rb") as f1, open(r2["pdf"], "rb") as f2:
        assert f1.read() == f2.read()


# --- daily removed ---------------------------------------------------------

def test_generate_daily_report_removed():
    assert not hasattr(rg, "generate_daily_report")

def test_period_bounds_daily_removed():
    assert not hasattr(rg, "_period_bounds")

def test_render_html_removed():
    assert not hasattr(rg, "render_html")


# --- dashboard links -------------------------------------------------------

def test_dashboard_has_no_daily_report_link():
    with open(DASHBOARD, "r", encoding="utf-8") as fh:
        html = fh.read()
    assert "reports/daily" not in html
    assert "Daily report" not in html
    assert "reports/monthly/latest.pdf" in html

def test_dashboard_keeps_monthly_links():
    with open(DASHBOARD, "r", encoding="utf-8") as fh:
        html = fh.read()
    assert "reports/monthly/latest.pdf" in html
    assert "Open Monthly PDF" in html or "Monthly PDF" in html
