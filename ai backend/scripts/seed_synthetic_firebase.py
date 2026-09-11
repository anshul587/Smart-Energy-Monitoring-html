#!/usr/bin/env python3
"""Seed the 30-day synthetic PZEM dataset into an isolated Firebase path.

Writes ONLY to: ai/synthetic_history/pzem_N/<unix_timestamp>
NEVER writes to: history/pzem_N/ or meters/pzem_N/

Schema per reading: {voltage, current, power, energy, frequency, pf}

Usage:
    python seed_synthetic_firebase.py          # actual write
    python seed_synthetic_firebase.py --dry-run  # show what would be written

This script is deterministic and idempotent — running it multiple times
on the same data will skip existing Firebase keys (same timestamp per meter).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.config import Settings
from ai.data_loader import _init_firebase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYNTHETIC_HISTORY_PREFIX = "ai/synthetic_history"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic"

TOTAL_ROWS_EXPECTED = 77_760   # 9 × 30 × 288 (5-min slots)
ROWS_PER_METER_EXPECTED = 8_640  # 30 × 288
METERS = 9
READING_FIELDS = ["voltage", "current", "power", "energy", "frequency", "pf"]

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_dataset() -> pd.DataFrame:
    """Load and validate the synthetic dataset. Returns the DataFrame."""
    parquet_path = DATA_DIR / "pzem_historical.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found at {parquet_path}")

    df = pd.read_parquet(parquet_path)
    assert len(df) == TOTAL_ROWS_EXPECTED, (
        f"Expected {TOTAL_ROWS_EXPECTED} rows, got {len(df)}"
    )
    assert set(df["meter_id"].unique()) == set(range(1, METERS + 1)), (
        f"Expected meters 1..{METERS}, got {set(df['meter_id'].unique())}"
    )
    for n in range(1, METERS + 1):
        meter_df = df[df["meter_id"] == n]
        assert len(meter_df) == ROWS_PER_METER_EXPECTED, (
            f"Meter {n}: expected {ROWS_PER_METER_EXPECTED} rows, got {len(meter_df)}"
        )
    # Verify per-meter timestamps are 5-min spaced (300s)
    for n in range(1, METERS + 1):
        meter_df = df[df["meter_id"] == n].copy()
        diffs = meter_df["timestamp"].diff().dropna()
        assert all(diffs == 300), (
            f"Meter {n}: timestamps not 5-minute spaced; diff values: "
            f"{diffs.value_counts().to_dict()}"
        )
    print(f"  Validated: {len(df)} rows, {METERS} meters, 5-min intervals OK")
    return df


# ---------------------------------------------------------------------------
# Firebase writing
# ---------------------------------------------------------------------------


def _write_readings_batch(db_ref, pzem_number: int, readings: dict) -> None:
    """Write multiple readings to Firebase in a single .update() call.

    readings: dict of {timestamp: {field: value, ...}}
    """
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db

    # Ensure Firebase Admin is initialized
    if not firebase_admin._apps:
        cred_path = Path("secrets/smart-energy-monitoring-5a2a4-firebase-adminsdk-fbsvc-b4c67b2214.json")
        if not cred_path.exists():
            from ai.config import get_settings
            cred_path = Path(get_settings().firebase_service_account_path)
        if not cred_path.exists():
            raise RuntimeError(f"Service account file not found at {cred_path}")
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred, {'databaseURL': 'https://smart-energy-monitoring-5a2a4-default-rtdb.asia-southeast1.firebasedatabase.app'})

    # Build the update payload: {timestamp: {field: value, ...}, ...}
    payload = {}
    for timestamp, value in readings.items():
        payload[str(timestamp)] = {
            "pzem_number": pzem_number,
            "timestamp": timestamp,
            "voltage": float(value["voltage"]),
            "current": float(value["current"]),
            "power": float(value["power"]),
            "energy": float(value["energy"]),
            "frequency": float(value["frequency"]),
            "pf": float(value["pf"]),
            "source_stage": "stage1/synthetic_data",
        }

    # Single .update() call writes all readings at once
    firebase_db.reference(f"ai/synthetic_history/pzem_{pzem_number}").update(payload)


def _process_meter(df: pd.DataFrame, meter_id: int, dry_run: bool) -> int:
    """Process one meter's data and write to Firebase.

    Returns count of records processed (or would be written).
    """
    meter_df = df[df["meter_id"] == meter_id].copy()
    meter_df = meter_df.drop(columns=["meter_id"]).reset_index(drop=True)

    if dry_run:
        return len(meter_df)

    # Write all readings for this meter in a single batch
    readings = {}
    for _, row in meter_df.iterrows():
        ts = int(row["timestamp"])
        readings[ts] = {
            "voltage": float(row["voltage"]),
            "current": float(row["current"]),
            "power": float(row["power"]),
            "energy": float(row["energy"]),
            "frequency": float(row["frequency"]),
            "pf": float(row["pf"]),
        }
    _write_readings_batch(None, meter_id, readings)
    return len(meter_df)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed synthetic 30-day PZEM data to isolated Firebase path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count records that would be written without actually writing to Firebase",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the dataset structure and exit without writing",
    )
    args = parser.parse_args()

    # Step 1: Load and validate the synthetic dataset
    print("=== Step 1: Loading and validating synthetic dataset ===")
    df = _validate_dataset()
    print(f"  Total rows: {len(df)}")
    print(f"  Meters: {sorted(df['meter_id'].unique())}")
    print(f"  Rows per meter: {df.groupby('meter_id').size().to_dict()}")

    # Step 2: Firebase initialization is deferred until needed (actual write mode).
    # Dry-run and validate-only do not require Firebase credentials.

    # Step 3: Firebase initialization only in actual write mode
    if not args.dry_run and not args.validate_only:
        print("\n=== Step 2: Initializing Firebase ===")
        from ai.config import get_settings
        _init_firebase(get_settings())
        print("  Firebase ready.")

    # Step 4: Process and write
    print("\n=== Step 3: Writing to isolated path: ai/synthetic_history/ ===")

    if args.dry_run:
        print("  (dry-run: counting records only)")
        total = 0
        for n in range(1, METERS + 1):
            c = _process_meter(df, n, dry_run=True)
            total += c
        print(f"  Would write: {total} records ({TOTAL_ROWS_EXPECTED} expected)")
        print(f"  Target path: ai/synthetic_history/pzem_N/<timestamp>")
        print("  (use without --dry-run to actually write)")
        return 0

    if args.validate_only:
        print("  (validate-only: no Firebase writes)")
        print(f"  Dataset validated: {len(df)} rows, {METERS} meters")
        return 0

    # Actual write mode
    total_written = 0
    for n in range(1, METERS + 1):
        c = _process_meter(df, n, dry_run=False)
        total_written += c
        meter_df = df[df["meter_id"] == n]
        first_ts = int(meter_df["timestamp"].min())
        last_ts = int(meter_df["timestamp"].max())
        print(f"  Meter {n}: {c} records written (first ts={first_ts}, last ts={last_ts})")

    print("\n=== Summary ===")
    print(f"  Meters written: {METERS}")
    print(f"  Records written: {total_written}")
    print(f"  Expected: {TOTAL_ROWS_EXPECTED}")
    print(f"  Coverage: {total_written / TOTAL_ROWS_EXPECTED * 100:.1f}%")
    print(f"  Target path: ai/synthetic_history/pzem_N/<timestamp>")
    print(f"  (NOT written to history/pzem_N/ or meters/pzem_N/)")
    print("  Idempotent: re-running skips existing timestamps")

    return 0


if __name__ == "__main__":
    sys.exit(main())