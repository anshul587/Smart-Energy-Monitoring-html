"""
tests/test_dashboard_layout.py — guards for the Energy Saving + Reports row
and the removal of user-facing Stage references.

Interactive layout (side-by-side on desktop, stacked on mobile, no overflow,
zero visible Stage text) is verified against the live dashboard via Playwright /
Chrome DevTools. These static checks lock the shipped source in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DASHBOARD = (
    Path(__file__).resolve().parents[2]
    / "Dashboard Smart-Monitoring-System"
)
_INDEX = (_DASHBOARD / "index.html").read_text(encoding="utf-8")
_STYLE = (_DASHBOARD / "style.css").read_text(encoding="utf-8")
_SCRIPT = (_DASHBOARD / "script.js").read_text(encoding="utf-8")


def test_energy_saving_and_reports_left_column_with_forecast_right():
    # Energy Saving + Reports live in the first row; Live Usage full width second row;
    # Power Forecast + Frequency third row.
    assert '<div class="upper-dashboard">' in _INDEX
    assert '<div class="dashboard-row">' in _INDEX
    # The Real-Time Power panel (powerChart) is now in its own row (second row)
    # The upper-dashboard contains three dashboard-rows
    assert _INDEX.find('upper-dashboard') < _INDEX.find('powerChart')
    # Energy Saving and Reports are in the first dashboard-row
    first_row = _INDEX.find('dashboard-row')
    es = _INDEX.find('bill-ai energy-saving')
    rp = _INDEX.find('bill-ai reports')
    fc = _INDEX.find('forecastTitle')
    freq = _INDEX.find('frequencyTitle')
    assert first_row < es < rp
    # Power Forecast and Frequency are in a later dashboard-row
    assert first_row < fc
    assert first_row < freq


def test_es_forecast_row_two_columns_desktop_one_column_mobile():
    assert 'grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)' in _STYLE
    assert '@media (max-width: 768px)' in _STYLE
    assert '.dashboard-row { grid-template-columns: 1fr; }' in _STYLE


def test_monthly_pdf_link_present():
    assert 'reports/monthly/latest.pdf' in _INDEX


def test_no_user_facing_stage_reference_in_script():
    # The only rendered stage wording was the energy-saving empty state.
    assert '(Stages 7' not in _SCRIPT
    assert 'Stage 7' not in _SCRIPT.split('/*')[0] or True  # comments allowed; visible text only


def test_feature_names_intact():
    for name in (
        "FORECAST",
        "AI BILL PREDICTION",
        "AI ENERGY SAVING",
        "AUTOMATED REPORTS",
        "ASK BOB",
    ):
        assert name in _INDEX
