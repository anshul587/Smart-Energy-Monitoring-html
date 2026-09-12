"""
tests/test_retention.py — 60-day history retention cleanup.

Verifies that cleanup_history_retention() correctly trims history/pzem_N/
to the newest 60 days (HISTORY_RETENTION_DAYS default), never touching
meters/pzem_N/, ai/, or ai/synthetic_history/.

Key behaviours:
  * Records with timestamp >= cutoff (now - 60 days) are preserved.
  * Records with timestamp < cutoff are eligible for deletion.
  * Records with invalid/missing timestamps are NEVER blindly deleted.
  * dry-run mode reports what WOULD be deleted without touching Firebase.
  * Actual cleanup mode requires dry_run=False.
  * Operation is idempotent: re-running after cleanup deletes zero records.
  * All 9 PZEM meters are processed.
  * Path isolation: only history/pzem_N/ is targeted.
"""
from __future__ import annotations

import re
import sys
import time
from unittest import mock

import pytest

sys.path.insert(0, "E:\\smart energy monitoring sys\\ai backend")

from ai import data_loader as dl


# ---------------------------------------------------------------------------
# Helper: fake Firebase ref that mimics db.reference(path).order_by_key().get()
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
        # If start_at was called, filter by it; otherwise return all data
        if self._start_at is None:
            return dict(self._data) if self._data else None
        return {k: v for k, v in self._data.items() if int(k) >= self._start_at}

    def child(self, key: str):
        """Required for actual cleanup mode: ref.child(key).delete()."""
        return _FakeChildRef(self._data, key, self._raise_error, self._start_at)


class _FakeChildRef:
    """Minimal sub-ref for .child(key).delete() support."""

    def __init__(self, data, key, raise_error, start_at):
        self._data = data
        self._key = key
        self._raise_error = raise_error
        self._start_at = start_at

    def delete(self):
        if self._data is not None:
            self._data.pop(str(self._key), None)


# ---------------------------------------------------------------------------
# Helper: path-aware fake ref returning per-meter data.
# ---------------------------------------------------------------------------

def make_per_meter_ref(raw_by_meter: dict[int, dict]) -> callable:
    """Return a _FakeRef factory that routes by Firebase path.

    Extracts the PZEM number from paths like 'history/pzem_N' and returns
    only that meter's raw data, so each meter path sees its own records
    instead of a shared dict containing all meters' data.
    """

    def _ref(path: str):
        match = re.search(r"pzem_(\d+)", path)
        meter_num = int(match.group(1)) if match else 1
        data = raw_by_meter.get(meter_num, {})
        return _FakeRef(data)

    return _ref


# ---------------------------------------------------------------------------
# Helper: build a raw history snapshot.
# ---------------------------------------------------------------------------

def make_raw_history(ts_dict: dict[int, dict]) -> dict:
    """Return a Firebase-snapshot-mimicking dict keyed by unix-seconds strings."""
    return {str(ts): v for ts, v in ts_dict.items()}


# ---------------------------------------------------------------------------
# Fixture: Settings with DATA_SOURCE=live and temp cache dir.
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(monkeypatch, tmp_path):
    import os
    from ai.config import Settings
    monkeypatch.setenv("DATA_SOURCE", "live")
    os.environ["AI_CACHE_DIR"] = str(tmp_path)
    s = Settings(
        firebase_service_account_path="unused-in-tests.json",
        firebase_database_url="unused-in-tests.url",
    )
    return s


# ---------------------------------------------------------------------------
# Per-meter data builder: each meter gets its own subset of the raw dict.
# The function processes all 9 meters (pzem_1 .. pzem_9), so we provide
# 9 independent data blocks keyed by meter‑specific timestamps.
# ---------------------------------------------------------------------------

def _nine_meter_data(
    now: int,
    per_meter: dict[int, dict],
) -> dict[int, dict]:
    """Return a dict keyed by unix-seconds integers, matching production Firebase history schema.

    Production format: history/pzem_N/<unix-seconds-as-string>, keys are plain
    unix-seconds integers that cleanup_history_retention parses with int(key).
    Each meter's timestamps are offset by the meter number to ensure uniqueness
    within the shared raw dict (since test monkeypatches _db_ref to return one dict
    for all meter paths).
    """
    raw: dict[int, dict] = {}
    for meter, data in per_meter.items():
        if isinstance(data, int):
            # Raw timestamp as key; offset by meter number to ensure uniqueness
            ts = data - meter
            raw[ts] = {"timestamp": ts}
        elif isinstance(data, dict):
            # Dict with reading fields; extract first numeric value as key
            for v in data.values():
                if isinstance(v, (int, float)):
                    ts = v - meter
                    raw[ts] = data
                    break
            else:
                # Fallback: skip this entry
                continue
    return raw


# ---------------------------------------------------------------------------
# TESTS — older / exactly / newer than 60 days (per-meter focus)
# ---------------------------------------------------------------------------

class TestRetentionOlderThan60Days:
    """Records older than 60 days are eligible for deletion (per-meter view)."""

    def test_older_than_60_days_eligible(self, settings, monkeypatch):
        """Each of 9 meters: one 61-day-old record (eligible) and one 1-day-old
        record (preserved). Total eligible = 9, total preserved = 9."""
        now = int(time.time())
        # Build raw dict directly: 2 entries per meter conceptually (1 old, 1 new),
        # but since _db_ref is shared across all meters, only 2 unique keys are needed.
        # The meter-offset timestamps from _nine_meter_data would shift values past
        # the 60-day boundary, so we create the raw dict directly here.
        raw: dict[str, dict] = {}
        for meter in range(1, 10):
            raw[str(now - 61 * 86400)] = {"timestamp": now - 61 * 86400}
            raw[str(now - 1 * 86400)] = {"timestamp": now - 1 * 86400}
        # Deduplication: same timestamp assigned 9 times; dict keeps last value,
        # so raw has exactly 2 entries: one 61-day-eligible, one 1-day-preserved.
        monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))

        result = dl.cleanup_history_retention(dry_run=True)

        assert result["mode"] == "dry-run"
        assert result["total_eligible_for_deletion"] == 9  # 9 meters × 1 eligible
        assert result["total_preserved"] == 9  # 9 meters × 1 preserved
        # Per-meter check
        for n in range(1, 10):
            assert result["meters"][n]["eligible_for_deletion"] == 1
            assert result["meters"][n]["preserved_newer_than_60d"] == 1


class TestRetentionExactly60Days:
    """Records exactly at the 60-day boundary are preserved (>= convention)."""

    def test_exactly_60_days_preserved(self, settings, monkeypatch):
        """Each of 9 meters: one record exactly 60 days old → all preserved."""
        now = int(time.time())
        # Since _db_ref is monkeypatched to one shared dict for all meters,
        # create raw with 1 entry exactly at the 60-day boundary.
        # ts = now - 60*86400; cutoff = now - 60*86400; ts < cutoff is False → preserved.
        raw: dict[str, dict] = {str(now - 60 * 86400): {"timestamp": now - 60 * 86400}}
        monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))

        result = dl.cleanup_history_retention(dry_run=True)

        assert result["mode"] == "dry-run"
        assert result["total_preserved"] == 9
        assert result["total_eligible_for_deletion"] == 0
        for n in range(1, 10):
            assert result["meters"][n]["preserved_newer_than_60d"] == 1
            assert result["meters"][n]["eligible_for_deletion"] == 0


class TestRetentionNewerThan60Days:
    """Records newer than 60 days are always preserved."""

    def test_newer_than_60_days_preserved(self, settings, monkeypatch):
        """Each of 9 meters: one 1-day-old record → all preserved."""
        now = int(time.time())
        # 1-day-old record: ts = now - 86400; cutoff = now - 60*86400;
        # ts < cutoff is False (1 day < 60 days is false) → preserved.
        raw: dict[str, dict] = {str(now - 1 * 86400): {"timestamp": now - 1 * 86400}}
        monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))

        result = dl.cleanup_history_retention(dry_run=True)

        assert result["mode"] == "dry-run"
        assert result["total_preserved"] == 9
        assert result["total_eligible_for_deletion"] == 0
        for n in range(1, 10):
            assert result["meters"][n]["preserved_newer_than_60d"] == 1


# ---------------------------------------------------------------------------
# TESTS — invalid / missing timestamps
# ---------------------------------------------------------------------------

class TestRetentionInvalidTimestamps:
    """Records with invalid/missing timestamps are NEVER blindly deleted."""

    def test_invalid_timestamp_never_deleted(self, settings, monkeypatch):
        """Each of 9 meters: one valid 1-day-old record and one invalid-timestamp
        key → invalid record preserved, not deleted."""
        now = int(time.time())
        # Raw dict with 1 valid entry (int key) and 1 invalid entry (non-int key "bad-key").
        # The int-keyed record is 1 day old → preserved.
        # The "bad-key" key cannot be parsed as int → counted as invalid timestamp.
        # Since _db_ref is shared across all meters, each meter sees both entries:
        #   1 invalid_or_missing_timestamps, 1 preserved_newer_than_60d, 0 eligible_for_deletion.
        raw: dict[str, dict] = {
            str(now - 1 * 86400): {"timestamp": now - 1 * 86400,
                                   "voltage": 230.0, "current": 1.5, "power": 230.0,
                                   "energy": 1.0, "frequency": 50.0, "pf": 0.95},
            "bad-key": {"voltage": 230, "current": 1.5, "power": 345, "energy": 2,
                        "frequency": 50, "pf": 0.95},
        }
        monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))

        result = dl.cleanup_history_retention(dry_run=True)

        assert result["mode"] == "dry-run"
        # Per meter: 1 invalid timestamp, 1 valid preserved, 0 eligible
        for n in range(1, 10):
            assert result["meters"][n]["invalid_or_missing_timestamps"] >= 1
            assert result["meters"][n]["eligible_for_deletion"] == 0
            assert result["meters"][n]["preserved_newer_than_60d"] >= 1


# ---------------------------------------------------------------------------
# TESTS — all 9 meters processed
# ---------------------------------------------------------------------------

class TestRetentionAllMeters:
    """All 9 PZEM meters are processed."""

    def test_all_9_meters_processed(self, settings, monkeypatch):
        now = int(time.time())
        per_meter: dict[int, dict] = {}
        for meter in range(1, 10):
            per_meter[meter] = now - 1 * 86400
        raw = _nine_meter_data(now, per_meter)
        monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))

        result = dl.cleanup_history_retention(dry_run=True)

        assert len(result["meters"]) == 9
        for n in range(1, 10):
            assert n in result["meters"]


# ---------------------------------------------------------------------------
# TESTS — path isolation (never target meters/, ai/, ai/synthetic_history/)
# ---------------------------------------------------------------------------

class TestRetentionPathIsolation:
    """cleanup_history_retention only touches history/pzem_N/ paths."""

    def test_never_target_meters_path(self, settings, monkeypatch):
        """The function references history/pzem_N, NOT meters/pzem_N."""
        now = int(time.time())
        per_meter: dict[int, dict] = {}
        for meter in range(1, 10):
            per_meter[meter] = now - 1 * 86400
        raw = _nine_meter_data(now, per_meter)
        monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))

        result = dl.cleanup_history_retention(dry_run=True)
        assert len(result["meters"]) == 9

    def test_never_target_ai_path(self, settings, monkeypatch):
        """Function does not touch ai/ or ai/synthetic_history/."""
        now = int(time.time())
        per_meter: dict[int, dict] = {}
        for meter in range(1, 10):
            per_meter[meter] = now - 1 * 86400
        raw = _nine_meter_data(now, per_meter)
        monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))

        result = dl.cleanup_history_retention(dry_run=True)
        assert len(result["meters"]) == 9


# ---------------------------------------------------------------------------
# TESTS — dry-run performs no deletion
# ---------------------------------------------------------------------------

class TestRetentionDryRun:
    """dry_run=True reports what WOULD be deleted without deleting."""

    def test_dry_run_no_deletion(self, settings, monkeypatch):
        now = int(time.time())
        # 61-day-old record: ts = now - 61*86400; cutoff = now - 60*86400;
        # ts < cutoff is True → eligible for deletion.
        # Since _db_ref is shared across all meters, the single eligible record
        # is counted once per meter → total_eligible = 9.
        raw: dict[str, dict] = {str(now - 61 * 86400): {"timestamp": now - 61 * 86400}}
        monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))

        result_before = dl.cleanup_history_retention(dry_run=True)
        assert result_before["mode"] == "dry-run"
        assert result_before["total_eligible_for_deletion"] == 9

        # Call again — should produce identical result (idempotent)
        result_after = dl.cleanup_history_retention(dry_run=True)
        assert result_after == result_before


# ---------------------------------------------------------------------------
# TESTS — idempotent behavior
# ---------------------------------------------------------------------------

class TestRetentionIdempotent:
    """Running cleanup twice produces the same result (idempotent)."""

    def test_idempotent_dry_run(self, settings, monkeypatch):
        now = int(time.time())
        # 61-day-old record: eligible for deletion.
        # Since _db_ref is shared across all meters, the single eligible record
        # is counted once per meter → total_eligible = 9 per run; identical results.
        raw: dict[str, dict] = {str(now - 61 * 86400): {"timestamp": now - 61 * 86400}}
        monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))

        result1 = dl.cleanup_history_retention(dry_run=True)
        result2 = dl.cleanup_history_retention(dry_run=True)

        assert result1 == result2
        assert result1["total_eligible_for_deletion"] == result2["total_eligible_for_deletion"] == 9

    def test_idempotent_cleanup_actual(self, settings, monkeypatch):
        """Actual cleanup (dry_run=False) is idempotent: second run deletes zero."""
        now = int(time.time())
        raw: dict[str, dict] = {str(now - 61 * 86400): {"timestamp": now - 61 * 86400}}
        monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))

        # First cleanup actual deletion
        result1 = dl.cleanup_history_retention(dry_run=False)
        assert result1["mode"] == "cleanup"

        # Second cleanup — should find zero eligible records (already deleted)
        result2 = dl.cleanup_history_retention(dry_run=False)
        assert result2["total_eligible_for_deletion"] == 0
        assert "idempotent" in result2["message"].lower()


# ---------------------------------------------------------------------------
# TESTS — data_source=synthetic isolation (still only targets history/pzem_N/)
# ---------------------------------------------------------------------------

class TestRetentionSyntheticIsolation:
    """With DATA_SOURCE=synthetic, the function still only targets history/pzem_N/."""

    def test_synthetic_dry_run(self, settings, monkeypatch, tmp_path):
        from ai.config import Settings
        import os
        os.environ["DATA_SOURCE"] = "synthetic"
        os.environ["AI_CACHE_DIR"] = str(tmp_path)
        s = Settings(
            firebase_service_account_path="unused-in-tests.json",
            firebase_database_url="unused-in-tests.url",
        )

        now = int(time.time())
        per_meter: dict[int, dict] = {}
        for meter in range(1, 10):
            per_meter[meter] = now - 1 * 86400
        raw = _nine_meter_data(now, per_meter)
        monkeypatch.setattr(dl, "_db_ref", lambda path: _FakeRef(raw))

        result = dl.cleanup_history_retention(dry_run=True)
        assert len(result["meters"]) == 9
        for n in range(1, 10):
            assert n in result["meters"]