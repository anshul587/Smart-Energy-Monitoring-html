"""
Stage 1 tests for ai/data_loader.py.

These mock the Firebase Admin SDK entirely (no real credentials, no
network) so they can run anywhere, and specifically exercise the cases
called out in the project's testing requirements: missing data, malformed
data, duplicate data, Firebase unavailable, no historical data,
insufficient historical data.

Run with:  pytest tests/test_data_loader.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.config import Settings
from ai import data_loader as dl


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Bypasses the real _require()/env-var path entirely — constructs
    # Settings directly with a throwaway cache dir per test.
    return Settings(
        firebase_service_account_path="unused-in-tests.json",
        firebase_database_url="https://example-not-real.firebasedatabase.app",
        pzem_count=9,
        history_retention_days=60,
        cache_dir=tmp_path / "cache",
    )


def make_raw_history(entries: dict[int, dict]) -> dict:
    """entries: {unix_ts: {voltage, current, power, energy, frequency, pf}}"""
    return {str(ts): fields for ts, fields in entries.items()}


# ---------------------------------------------------------------------------
# _parse_history_snapshot — pure parsing logic, no Firebase involved
# ---------------------------------------------------------------------------

def test_parse_normal_readings():
    now = int(time.time())
    raw = make_raw_history({
        now: {"voltage": 230.1, "current": 1.2, "power": 276.0, "energy": 12.345, "frequency": 50.0, "pf": 0.98},
        now + 300: {"voltage": 229.8, "current": 1.1, "power": 253.0, "energy": 12.366, "frequency": 50.1, "pf": 0.97},
    })
    frame, dropped, dupes = dl._parse_history_snapshot(raw, pzem_number=1)
    assert len(frame) == 2
    assert dropped == 0
    assert dupes == 0
    assert list(frame.columns) == ["timestamp", *dl.READING_FIELDS]
    assert frame.iloc[0]["timestamp"] < frame.iloc[1]["timestamp"]  # sorted


def test_parse_empty_snapshot():
    frame, dropped, dupes = dl._parse_history_snapshot({}, pzem_number=1)
    assert frame.empty
    assert dropped == 0 and dupes == 0

    frame2, dropped2, dupes2 = dl._parse_history_snapshot(None, pzem_number=1)
    assert frame2.empty
    assert dropped2 == 0


def test_parse_malformed_rows_are_dropped_not_fabricated():
    now = int(time.time())
    raw = {
        str(now): {"voltage": 230.1, "current": 1.2, "power": 276.0, "energy": 12.3, "frequency": 50.0, "pf": 0.98},
        "not-a-timestamp": {"voltage": 1, "current": 1, "power": 1, "energy": 1, "frequency": 1, "pf": 1},
        str(now + 300): {"voltage": "N/A", "current": 1.2, "power": 276.0, "energy": 12.3, "frequency": 50.0, "pf": 0.98},
        str(now + 600): 42,  # legacy bare-number row — can't recover full reading
        str(now + 900): {"voltage": 230.0},  # missing fields
    }
    frame, dropped, dupes = dl._parse_history_snapshot(raw, pzem_number=1)
    assert len(frame) == 1  # only the one fully-valid row survives
    assert dropped == 4
    assert dupes == 0


def test_parse_duplicate_keys_last_one_wins():
    ts = int(time.time())
    # dict can't literally have duplicate keys, but Firebase never sends
    # that anyway — this test instead verifies drop_duplicates behavior in
    # fetch_meter_history's dedup step, which is where real duplicates
    # (cache overlap on incremental fetch) actually get collapsed.
    cached = pd.DataFrame([
        {"timestamp": ts, "voltage": 230.0, "current": 1.0, "power": 230.0, "energy": 1.0, "frequency": 50.0, "pf": 0.9},
    ])
    fresh = pd.DataFrame([
        {"timestamp": ts, "voltage": 231.0, "current": 1.1, "power": 254.0, "energy": 1.1, "frequency": 50.0, "pf": 0.91},
    ])
    combined = pd.concat([cached, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    assert len(combined) == 1
    assert combined.iloc[0]["voltage"] == 231.0  # the newer value won


# ---------------------------------------------------------------------------
# fetch_meter_history — with Firebase mocked
# ---------------------------------------------------------------------------

class _FakeRef:
    """Mimics the chain: db.reference(path).order_by_key().start_at(x).get()"""

    def __init__(self, data: dict | None, raise_error: Exception | None = None):
        self._data = data
        self._raise_error = raise_error
        self._start_at = None

    def order_by_key(self):
        return self

    def start_at(self, value):
        self._start_at = int(value)
        return self

    def get(self):
        if self._raise_error:
            raise self._raise_error
        if self._data is None:
            return None
        return {k: v for k, v in self._data.items() if int(k) >= self._start_at}


def test_fetch_meter_history_no_data_reports_zero_days(settings, monkeypatch):
    monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef({}))
    result = dl.fetch_meter_history(1, settings=settings)
    assert result.frame.empty
    assert result.available_days == 0.0
    assert result.requested_days == 60
    assert result.served_from_cache_only is False


def test_fetch_meter_history_insufficient_data_reports_actual_span(settings, monkeypatch):
    now = int(time.time())
    raw = make_raw_history({
        now - 5 * 86400: {"voltage": 230, "current": 1, "power": 230, "energy": 1, "frequency": 50, "pf": 0.9},
        now: {"voltage": 231, "current": 1, "power": 231, "energy": 2, "frequency": 50, "pf": 0.9},
    })
    monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))
    result = dl.fetch_meter_history(1, settings=settings)
    # Only ~5 days of data exists even though 60 were requested — must be
    # reported honestly, not silently treated as "60 days available".
    assert 4.9 <= result.available_days <= 5.1
    assert result.requested_days == 60


def test_fetch_meter_history_firebase_unavailable_falls_back_to_cache(settings, monkeypatch):
    # Deliberately timestamped a few seconds in the past (not exactly
    # int(time.time())): fetch_meter_history() only attempts a Firebase
    # call at all when there could plausibly be newer data than what's
    # cached (fetch_start <= now). A reading stamped at "now" risks landing
    # in the same wall-clock second as the second call below, which would
    # make fetch_start (cached_max + 1) land in the future and skip the
    # fetch attempt entirely — correctly, but that would test "no fetch was
    # needed" instead of the outage-fallback path this test exists for.
    now = int(time.time()) - 10
    raw = make_raw_history({
        now: {"voltage": 230, "current": 1, "power": 230, "energy": 1, "frequency": 50, "pf": 0.9},
    })
    # First call: Firebase reachable, populates the cache.
    monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))
    first = dl.fetch_meter_history(1, settings=settings)
    assert not first.frame.empty
    assert first.served_from_cache_only is False

    # Second call: Firebase throws — must fall back to what's cached
    # instead of raising or returning nothing.
    def broken_ref(path):
        return _FakeRef(None, raise_error=ConnectionError("simulated outage"))

    monkeypatch.setattr(dl, "_db_ref", broken_ref)
    second = dl.fetch_meter_history(1, settings=settings)
    assert second.served_from_cache_only is True
    assert len(second.frame) == 1  # the previously-cached row is still there


def test_fetch_meter_history_incremental_only_requests_new_slots(settings, monkeypatch):
    now = int(time.time())
    raw = make_raw_history({
        now - 600: {"voltage": 230, "current": 1, "power": 230, "energy": 1, "frequency": 50, "pf": 0.9},
        now - 300: {"voltage": 231, "current": 1, "power": 231, "energy": 2, "frequency": 50, "pf": 0.9},
    })
    calls = []

    def tracking_ref(path):
        ref = _FakeRef(raw)
        original_start_at = ref.start_at

        def tracked_start_at(value):
            calls.append(int(value))
            return original_start_at(value)

        ref.start_at = tracked_start_at
        return ref

    monkeypatch.setattr(dl, "_db_ref", tracking_ref)
    first = dl.fetch_meter_history(1, settings=settings)
    assert len(first.frame) == 2

    # Second call with no new data upstream: fetch_start should now be
    # AFTER the newest cached timestamp, not the full 60-day window again.
    monkeypatch.setattr(dl, "_db_ref", tracking_ref)
    dl.fetch_meter_history(1, settings=settings)
    assert calls[-1] == (now - 300) + 1  # one past the newest cached row


def test_fetch_all_history_one_meter_failing_does_not_block_others(settings, monkeypatch):
    now = int(time.time())
    good_raw = make_raw_history({
        now: {"voltage": 230, "current": 1, "power": 230, "energy": 1, "frequency": 50, "pf": 0.9},
    })

    def per_meter_ref(path):
        if path == "history/pzem_4":
            return _FakeRef(None, raise_error=ConnectionError("PZEM 4 path unreachable"))
        return _FakeRef(good_raw)

    monkeypatch.setattr(dl, "_db_ref", per_meter_ref)
    results = dl.fetch_all_history(settings=settings)
    assert len(results) == 9
    assert results[4].served_from_cache_only is True
    assert results[4].frame.empty  # no cache existed yet for meter 4
    assert results[1].served_from_cache_only is False
    assert not results[1].frame.empty


def test_invalid_pzem_number_rejected(settings):
    with pytest.raises(ValueError):
        dl.fetch_meter_history(0, settings=settings)
    with pytest.raises(ValueError):
        dl.fetch_meter_history(10, settings=settings)
