"""
ai/anomaly_detection.py
------------------------
STAGE 3 (revised): Operating-state-aware anomaly detection.

Takes the per-PZEM feature frames produced by ai.preprocessing (Stage 2)
and asks, independently for each meter:

    "Is this PZEM currently behaving unusually compared with its own
    historical ACTIVE-operation behavior?"

This module NEVER claims a component has failed. An Isolation Forest
flag means "statistically unusual compared to this meter's own recent
ACTIVE-state history" — nothing more. Fault diagnosis (deciding WHY
something is unusual, and how severe that is) is explicitly out of
scope; that is Stage 4's job. The `anomaly_severity_provisional` column
below exists only because Stage 4 needs *something* to build on — it is
a coarse, clearly-labeled placeholder, not a diagnosis.

===========================================================================
WHY THIS REVISION EXISTS — no fixed classroom schedule
===========================================================================
The classroom this deployment monitors has NO fixed operating window.
It may start early or late, stop early or run long, take breaks, or sit
unused for an entire day. Any anomaly-detection design that assumes a
clock-based "class is in session 11:00-18:00" schedule, or that assumes
"power > 0 means the classroom is active" (standby/idle loads are
non-zero), would silently misclassify normal variation as anomalous, or
train on data that mixes active and idle behavior and blur what "normal
active operation" actually looks like.

So this module now detects an ACTIVE/INACTIVE **operating state** for
every row using only the electrical measurements themselves (never
timestamp/hour), and trains Isolation Forest ONLY on rows the detector
believes represent genuine active operation. Time-of-day features
(hour_of_day, day_of_week, is_weekend) are still available to Isolation
Forest as ML inputs — a real shift in *when* the classroom tends to
operate can be informative — but they play NO role in deciding whether a
given row is active or idle. See `detect_operating_state()` below for
the full method and rationale.

One model per meter
--------------------
A separate IsolationForest is trained per PZEM, on that PZEM's own
ACTIVE-state history only. Different appliances/circuits have different
normal operating envelopes, so a single fleet-wide model would either be
too loose for quiet circuits or flag heavy circuits as constantly
anomalous. This module never pools rows across PZEMs for training, and
never pools ACTIVE and INACTIVE rows together for training.

Reuses, does not duplicate
---------------------------
This module does not touch Firebase and does not re-implement cleaning
or feature engineering — it consumes ai.preprocessing.PreprocessResult
(Stage 2 output) and, when one isn't supplied, calls
ai.preprocessing.preprocess_meter()/run_preprocessing_pipeline() itself,
the same way Stage 2 calls into Stage 1 rather than re-reading Firebase
paths directly. Stage 1 and Stage 2 are unmodified by this revision.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

from . import preprocessing
from .config import Settings, get_settings
from .preprocessing import PreprocessResult

logger = logging.getLogger("ai.anomaly_detection")

# ===========================================================================
# PART 1 — OPERATING-STATE DETECTION (ACTIVE vs INACTIVE)
# ===========================================================================
#
# METHOD, in one paragraph: for each PZEM independently, we fit a
# 2-component Gaussian Mixture Model on that meter's own (standardized)
# power and current readings. Electrical load behind a real on/off-ish
# circuit is naturally bimodal — a quiet/standby cluster of readings and
# a drawing-real-load cluster — so an *unsupervised* 2-component split
# discovers each meter's own two regimes directly from its own data,
# with no fixed watt threshold and no assumption that every appliance's
# "on" state looks the same. The component with the HIGHER mean power is
# labeled ACTIVE, the other INACTIVE. This is standard practice for
# on/off load disaggregation from power traces and is exactly the kind
# of "adaptive per PZEM" approach the project requires.
#
# We do not trust the GMM split blindly. Two guardrails:
#   1. MIN_STATE_COMPONENT_WEIGHT — if one component only explains a
#      tiny sliver of the data (e.g. a handful of noise outliers), that
#      is not evidence of a genuine second operating regime.
#   2. MIN_STATE_SEPARATION_RATIO — the two components' power means must
#      differ by at least this many pooled standard deviations, or the
#      "split" is likely just noise around a single true regime.
#
# If either guardrail fails (or the GMM fit itself fails, e.g. on
# near-constant data with no real variance to cluster), we fall back to
# a percentile-based rule that is still per-meter/adaptive (not a fixed
# absolute watt number): a meter with essentially no variability is not
# assumed to be idle just because we couldn't find two clusters — we
# assume it represents one operating regime and treat everything as
# ACTIVE, since defaulting to "assume idle" would silently discard a
# legitimately-active but low-variance circuit's entire history. A
# meter with real variability but no clean 2-cluster split uses its own
# 10th percentile of power as an idle/active dividing line — still
# derived per-meter from that meter's own distribution, not a shared
# constant.
#
# PERSISTENCE / DEBOUNCE: a single noisy sample must not flip the
# operating state. After the per-row ACTIVE/INACTIVE call above, we
# collapse any run of consecutive same-state rows shorter than
# `state_persistence_samples` into whatever state surrounds it — the
# same idea as a debounce filter on a physical switch. Default is 2
# consecutive 5-minute samples (10 minutes): short enough not to erase a
# genuine brief class changeover, long enough that one glitchy reading
# can't register as a full state transition. This is configurable
# because a deployment with noisier PZEM readings, or one where very
# short real transitions matter, may want a different value.

# Signals used to separate operating states. Only power/current — not
# voltage, pf, or frequency, which primarily reflect grid quality and
# don't distinguish "drawing load" from "idle" the way power/current do.
STATE_FEATURES: list[str] = ["power", "current"]

# See guardrail (1) above. 2% is deliberately loose: a real idle period
# in a classroom that runs mostly-active days could well be a small
# minority of a 60-day window, and we don't want to reject a genuine
# (if infrequent) idle cluster just because it's the smaller one.
MIN_STATE_COMPONENT_WEIGHT = 0.02

# See guardrail (2) above. A ratio of 1.0 means the two candidate
# regimes' power means differ by at least one pooled standard deviation
# — a conservative but not extreme bar for "these are probably two real
# regimes, not one noisy one."
MIN_STATE_SEPARATION_RATIO = 1.0

# Used only by the percentile fallback, and only when the GMM split is
# rejected by the guardrails above. Per-meter (computed from that
# meter's own power distribution each time), so this is not a shared
# fixed watt number across the fleet.
FALLBACK_IDLE_PERCENTILE = 10.0

# Below this coefficient of variation in power, we treat the meter as
# having no meaningful separable variability at all and assume ACTIVE
# throughout, rather than guessing at an idle split from noise.
FALLBACK_MIN_COEFFICIENT_OF_VARIATION = 0.01

# Debounce window, in consecutive 5-minute samples. See rationale above.
DEFAULT_STATE_PERSISTENCE_SAMPLES = 2

# GMM fit is randomized (k-means++-style init); fixed for reproducible
# state calls across runs on the same data.
DEFAULT_STATE_RANDOM_STATE = 42


@dataclass
class OperatingStateResult:
    """Per-meter operating-state detection output. `labels` is aligned
    1:1 with the input frame's row order (ACTIVE/INACTIVE per row,
    post-persistence). The rest is metadata for transparency/debugging
    and for the real-Firebase report — never used to compute anomaly
    scores itself."""

    labels: np.ndarray             # dtype=object, "ACTIVE" / "INACTIVE", one per input row
    method: str                    # "gmm" | "fallback_percentile" | "fallback_no_variability"
    active_rows: int
    inactive_rows: int
    active_component_mean_power: Optional[float]
    inactive_component_mean_power: Optional[float]
    separation_ratio: Optional[float]
    detail: str                    # human-readable explanation of what happened and why


def _apply_persistence(labels: np.ndarray, min_persistence: int) -> np.ndarray:
    """Collapses any run of consecutive identical labels shorter than
    `min_persistence` into the label of whatever precedes it (or, if the
    short run is at the very start of the series with nothing before it,
    into whatever follows). This is a simple debounce: it prevents a
    single (or handful of) noisy sample(s) from registering as a
    complete operating-state transition, without requiring any fixed
    clock schedule.

    min_persistence <= 1 disables debouncing entirely (every row is
    trusted as-is) — a legitimate configuration choice for very
    clean/low-noise PZEM data.
    """
    n = len(labels)
    if min_persistence <= 1 or n == 0:
        return labels.copy()

    out = labels.copy()
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        run_len = j - i
        if run_len < min_persistence:
            if i == 0:
                # No confirmed state yet to fall back on — borrow the
                # state of whatever comes right after this short run, if
                # anything does; otherwise there's nothing to debounce
                # against and the run stands.
                fallback = labels[j] if j < n else labels[i]
            else:
                fallback = out[i - 1]
            out[i:j] = fallback
        i = j
    return out


def detect_operating_state(
    df: pd.DataFrame,
    state_features: list[str] = STATE_FEATURES,
    persistence_samples: int = DEFAULT_STATE_PERSISTENCE_SAMPLES,
    min_separation_ratio: float = MIN_STATE_SEPARATION_RATIO,
    min_component_weight: float = MIN_STATE_COMPONENT_WEIGHT,
    fallback_idle_percentile: float = FALLBACK_IDLE_PERCENTILE,
    fallback_min_cv: float = FALLBACK_MIN_COEFFICIENT_OF_VARIATION,
    random_state: int = DEFAULT_STATE_RANDOM_STATE,
) -> OperatingStateResult:
    """Classifies every row of `df` (a Stage 2 feature frame for ONE
    PZEM, in timestamp order) as ACTIVE or INACTIVE. See the module-level
    comment block above for the full method and the reasoning behind
    each guardrail/fallback. Never touches timestamp/hour_of_day/
    day_of_week — operating state is derived from electrical behavior
    only.
    """
    n = len(df)
    if n == 0:
        return OperatingStateResult(
            labels=np.array([], dtype=object),
            method="empty",
            active_rows=0,
            inactive_rows=0,
            active_component_mean_power=None,
            inactive_component_mean_power=None,
            separation_ratio=None,
            detail="No rows to classify.",
        )

    available = [c for c in state_features if c in df.columns and df[c].notna().all()]
    power = df["power"].to_numpy(dtype="float64") if "power" in df.columns else None

    if power is None or not available:
        # Should not happen for a Stage-2 READY meter (power is always
        # present and clean), but we never assume state we can't
        # observe — treat as one undifferentiated active regime rather
        # than guessing.
        labels = np.full(n, "ACTIVE", dtype=object)
        return OperatingStateResult(
            labels=labels,
            method="no_state_features_available",
            active_rows=n,
            inactive_rows=0,
            active_component_mean_power=None,
            inactive_component_mean_power=None,
            separation_ratio=None,
            detail="power/current not available on this frame; assumed ACTIVE throughout.",
        )

    raw_labels: Optional[np.ndarray] = None
    method = ""
    detail = ""
    active_mean: Optional[float] = None
    inactive_mean: Optional[float] = None
    separation_ratio: Optional[float] = None

    # --- Primary method: 2-component GMM on standardized power/current ---
    X = df[available].to_numpy(dtype="float64")
    try:
        if not np.isfinite(X).all():
            # Defensive only: Stage 2 is supposed to guarantee finite
            # power/current for a READY meter. If that guarantee is ever
            # violated, don't let a stray inf/NaN silently poison the
            # mean/std used for clustering — fall through to the
            # percentile fallback below instead. The offending row is
            # excluded from training later regardless (see the
            # dropna/isfinite filter in detect_anomalies_for_meter).
            raise ValueError("non-finite value(s) present in state_features; skipping GMM split")
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma_safe = np.where(sigma > 0, sigma, 1.0)
        X_scaled = (X - mu) / sigma_safe

        gmm = GaussianMixture(n_components=2, random_state=random_state, n_init=3)
        component_assignment = gmm.fit_predict(X_scaled)

        weights = np.array([
            float(np.mean(component_assignment == c)) for c in (0, 1)
        ])
        cluster_power_means = np.array([
            float(power[component_assignment == c].mean()) if np.any(component_assignment == c) else np.nan
            for c in (0, 1)
        ])
        active_component = int(np.nanargmax(cluster_power_means))
        inactive_component = 1 - active_component

        pooled_std = float(power.std())
        if pooled_std > 0 and not np.isnan(cluster_power_means).any():
            separation = abs(cluster_power_means[active_component] - cluster_power_means[inactive_component])
            separation_ratio = separation / pooled_std
        else:
            separation_ratio = 0.0

        if weights.min() < min_component_weight:
            raise ValueError(
                f"smaller GMM component only accounts for "
                f"{weights.min():.4f} of rows (< {min_component_weight}); "
                f"not trusted as a genuine second operating regime"
            )
        if separation_ratio < min_separation_ratio:
            raise ValueError(
                f"GMM component power means separated by only "
                f"{separation_ratio:.2f}x pooled std (< {min_separation_ratio}x); "
                f"not trusted as two distinct regimes"
            )

        raw_labels = np.where(component_assignment == active_component, "ACTIVE", "INACTIVE")
        method = "gmm"
        active_mean = cluster_power_means[active_component]
        inactive_mean = cluster_power_means[inactive_component]
        detail = (
            f"2-component GMM on {available}: ACTIVE mean power="
            f"{active_mean:.2f}, INACTIVE mean power={inactive_mean:.2f}, "
            f"separation={separation_ratio:.2f}x pooled std, "
            f"weights={weights[active_component]:.3f}/{weights[inactive_component]:.3f}."
        )
    except Exception as exc:
        # Falls through to the percentile/constant fallback below. Not
        # logged as an error — an inconclusive split is an expected,
        # handled outcome for some meters (e.g. always-on circuits),
        # not a bug.
        logger.info("PZEM operating-state GMM split not used (%s); using fallback.", exc)
        raw_labels = None

    # --- Fallback: per-meter percentile, or "assume active" if no
    #     meaningful variability exists to split at all ---
    if raw_labels is None:
        finite_power_mask = np.isfinite(power)
        finite_power = power[finite_power_mask] if finite_power_mask.any() else power
        p_mean = float(finite_power.mean())
        p_std = float(finite_power.std())
        cv = (p_std / p_mean) if p_mean > 0 else 0.0
        p10 = float(np.percentile(finite_power, fallback_idle_percentile))

        if p_std == 0 or cv < fallback_min_cv or p10 >= float(finite_power.max()):
            raw_labels = np.full(n, "ACTIVE", dtype=object)
            method = "fallback_no_variability"
            detail = (
                f"No statistically meaningful power variability found "
                f"(coefficient of variation={cv:.4f}); rather than assume "
                f"idle, the entire series is treated as one ACTIVE regime."
            )
        else:
            raw_labels = np.where(power > p10, "ACTIVE", "INACTIVE")
            method = "fallback_percentile"
            detail = (
                f"No reliable 2-cluster split found; used this meter's own "
                f"{fallback_idle_percentile:.0f}th percentile of power "
                f"({p10:.2f} W) as an adaptive idle/active divider."
            )

    labels = _apply_persistence(raw_labels, persistence_samples)
    active_rows = int(np.sum(labels == "ACTIVE"))
    inactive_rows = n - active_rows

    return OperatingStateResult(
        labels=labels,
        method=method,
        active_rows=active_rows,
        inactive_rows=inactive_rows,
        active_component_mean_power=active_mean,
        inactive_component_mean_power=inactive_mean,
        separation_ratio=separation_ratio,
        detail=detail,
    )


# ===========================================================================
# PART 2 — MINIMUM ACTIVE TRAINING DATA
# ===========================================================================
#
# Replacing 288 (a fixed 24-hour assumption) with a data-driven pair of
# requirements, both configurable:
#
# MIN_ACTIVE_TRAINING_ROWS = 256
#   This is deliberately tied to scikit-learn's own IsolationForest
#   default: with max_samples="auto", each tree is built on
#   min(256, n_samples) rows drawn from the training set. Isolation
#   Forest's whole design premise is that *each tree only sees a
#   subsample*, so anomalies isolate in fewer splits than they would in
#   the full dataset. If we train on fewer than 256 rows, every tree
#   sees the ENTIRE training set every time — there is no subsampling
#   left to do, and the ensemble stops behaving the way the algorithm
#   was designed to. 256 is therefore not an arbitrary "feels like
#   enough" number the way the old 288 (calendar day) or a naive 84
#   (7 hours) would be — it's the point below which the model's own
#   subsampling stops being able to do anything.
#
# MIN_ACTIVE_TRAINING_DAYS = 3
#   256 active rows could, in principle, all come from one unusually
#   long single day (a variable-schedule classroom might run 20+ active
#   hours in one exam day). Training Isolation Forest on one day's
#   active behavior risks it learning "whatever happened that one day"
#   as the entire definition of normal — exactly the failure mode the
#   288-row fixed calendar day had, just relocated. Requiring active
#   rows to be drawn from at least 3 distinct calendar dates (UTC) means
#   the model has seen active-operation variation across at least a
#   handful of different days before it is trusted, without requiring
#   a full week (7 days) of history, which would leave the system unable
#   to detect anything for a new deployment for over a week.
#
# MIN_ACTIVE_ROWS_PER_DAY = 6 (30 minutes)
#   A day only "counts" toward the 3-distinct-days requirement if it
#   contributed at least this many confirmed-active rows. Otherwise a
#   single stray active sample on an otherwise-idle day (e.g. a false
#   positive from the state detector, or someone briefly turning on a
#   projector) would count as "a different operating day" and let a
#   dataset satisfy the day-diversity requirement without actually
#   representing meaningfully different operating sessions.
#
# None of these three numbers requires 24 consecutive hours, a fixed
# daily schedule, or a full week of data — they're satisfied by, for
# example, three days each with a couple of hours of class, in any
# pattern, on any days.

MIN_ACTIVE_TRAINING_ROWS = 256
MIN_ACTIVE_TRAINING_DAYS = 3
MIN_ACTIVE_ROWS_PER_DAY = 6

# IsolationForest's contamination parameter: the assumed fraction of
# training rows that are actually anomalous, which controls where the
# decision_function threshold between NORMAL/ANOMALY is drawn. 0.05 (5%)
# is scikit-learn's own historical default and a common starting
# heuristic in anomaly-detection literature — it is NOT derived from any
# measurement of this project's real fault rate, because no labeled
# fault data exists yet to measure it from. Treat this as a tunable
# knob, not a validated constant: if real operation shows most flags are
# false positives, lower it; if genuine anomalies are being missed,
# raise it.
DEFAULT_CONTAMINATION = 0.05

# Fixed by default so two runs on the same data produce the same
# NORMAL/ANOMALY calls — IsolationForest's tree construction is
# randomized. Override for an actual ensemble-variability study.
DEFAULT_RANDOM_STATE = 42

# scikit-learn's own default; called out explicitly (rather than left
# implicit) since it's the other lever, besides contamination, that
# meaningfully changes results.
DEFAULT_N_ESTIMATORS = 100

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

# Every column this stage is WILLING to use, in a fixed order (fixed
# order matters for reproducibility — sklearn sees columns positionally,
# not by name). Not every meter will actually have all of these
# available (see _select_features) — a meter with fewer than Stage 2's
# MIN_ROWS_FOR_LONG_WINDOW rows has NaN *_1d columns, for example — but
# nothing on this list is ever fabricated to fill a gap; unavailable
# columns are dropped from that meter's feature set entirely, not
# imputed. hour_of_day/day_of_week/is_weekend are retained here as ML
# INPUTS ONLY — they play no role in operating-state detection above.
CANDIDATE_FEATURES: list[str] = [
    # Raw readings
    "power",
    "current",
    "voltage",
    "pf",
    "frequency",
    # ~1h rolling statistics
    "rolling_mean_power_1h",
    "rolling_std_power_1h",
    "rolling_mean_current_1h",
    "rolling_std_current_1h",
    "rolling_mean_pf_1h",
    "rolling_std_pf_1h",
    # ~1d rolling statistics + trend
    "rolling_mean_power_1d",
    "rolling_std_power_1d",
    "rolling_trend_power_1d",
    "rolling_mean_current_1d",
    "rolling_std_current_1d",
    "rolling_trend_current_1d",
    "rolling_mean_pf_1d",
    "rolling_std_pf_1d",
    "rolling_trend_pf_1d",
    # Deviation from this meter's own baseline
    "deviation_power",
    "pct_deviation_power",
    "deviation_current",
    "pct_deviation_current",
    "deviation_pf",
    "pct_deviation_pf",
    # Time-of-use context (ML input only — NOT used to define active/idle)
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

# Columns Stage 2 produces that are deliberately EXCLUDED, and why. Kept
# here (rather than just in comments) so the reasoning is inspectable at
# runtime, e.g. from a REPL, not just readable in source.
EXCLUDED_STAGE2_COLUMNS: dict[str, str] = {
    "timestamp": (
        "Raw unix timestamp. Explicitly excluded per the Stage 3 spec: "
        "training on absolute time would let the model key off "
        "'this is a timestamp the training set didn't see' rather than "
        "genuine behavioral deviation. hour_of_day/day_of_week/is_weekend "
        "capture the meaningful time-of-use signal instead. timestamp is "
        "also never used to decide active vs. inactive operating state."
    ),
    "datetime_utc": "Not numeric; hour_of_day/day_of_week already derive from it.",
    "baseline_power": (
        "A single scalar (this meter's median power) broadcast identically "
        "onto every row -> zero variance -> no discriminative information "
        "for a row-level model. deviation_power / pct_deviation_power "
        "already encode the same reference relative to each row."
    ),
    "baseline_current": "Same rationale as baseline_power.",
    "baseline_pf": "Same rationale as baseline_power.",
}


def _select_features(df: pd.DataFrame) -> list[str]:
    """Returns the subset of CANDIDATE_FEATURES this particular meter's
    feature frame can actually support: present as a column AND not
    entirely NaN (e.g. the *_1d columns for a meter below Stage 2's
    MIN_ROWS_FOR_LONG_WINDOW). Order is preserved from CANDIDATE_FEATURES
    so results are reproducible run-to-run for the same meter."""
    selected = []
    for column in CANDIDATE_FEATURES:
        if column not in df.columns:
            continue
        if df[column].isna().all():
            continue
        selected.append(column)
    return selected


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AnomalyDetectionResult:
    """Everything a caller (report, future dashboard, Stage 4) needs for
    one PZEM. result_frame is None whenever model_status != "READY" —
    callers must check status before touching it, same convention as
    Stage 2's PreprocessResult.feature_frame.

    operating_state_* fields describe Stage 3's ACTIVE/INACTIVE call for
    this meter and are populated whenever operating-state detection ran
    at all (even if the model itself ends up INSUFFICIENT_DATA) — this
    keeps operating_state, anomaly_score, and model_status as clearly
    separate concepts, per the project's Stage 3 spec.
    """

    pzem_number: int
    model_status: str                 # "READY" or "INSUFFICIENT_DATA"
    reason: Optional[str]

    training_rows: int                # ACTIVE rows actually used to fit the model
                                       # (post feature-selection, post NaN/inf drop)
    features_used: list[str] = field(default_factory=list)
    contamination: Optional[float] = None
    random_state: Optional[int] = None

    # Operating-state detection metadata (see detect_operating_state()).
    operating_state_method: Optional[str] = None
    active_rows: int = 0               # total rows classified ACTIVE (pre feature-NaN drop)
    inactive_rows: int = 0             # total rows classified INACTIVE
    active_days_represented: int = 0   # distinct calendar days contributing >= MIN_ACTIVE_ROWS_PER_DAY

    # columns: pzem_id, timestamp, operating_state, anomaly_score,
    # anomaly_score_normalized, anomaly_label, anomaly_severity_provisional.
    # Includes ALL rows from the Stage 2 feature frame, in order — not
    # just the ones used for training. Only rows where operating_state ==
    # "ACTIVE" and features were complete/finite ever get a real score;
    # every other row has anomaly_label == "NOT_SCORED" and NaN scores.
    result_frame: Optional[pd.DataFrame] = field(default=None, repr=False)

    debug_traceback: Optional[str] = field(default=None, repr=False)


@dataclass
class FleetAnomalySummary:
    analyzed: int             # PZEMs considered (1..pzem_count)
    insufficient_data: int    # model_status == INSUFFICIENT_DATA
    anomalous_now: int        # READY meters whose MOST RECENT scored row is ANOMALY
    normal_now: int           # READY meters whose MOST RECENT scored row is NORMAL


def _insufficient_result(
    pzem_number: int,
    reason: str,
    training_rows: int = 0,
    features_used: Optional[list[str]] = None,
    debug_traceback: Optional[str] = None,
    operating_state_method: Optional[str] = None,
    active_rows: int = 0,
    inactive_rows: int = 0,
    active_days_represented: int = 0,
) -> AnomalyDetectionResult:
    """Shared constructor for every INSUFFICIENT_DATA / error path, so
    whatever was already known (e.g. which features would have been used,
    how many rows nearly made it, what operating-state detection found)
    is never silently dropped — same principle as
    preprocessing._insufficient_data_result."""
    return AnomalyDetectionResult(
        pzem_number=pzem_number,
        model_status="INSUFFICIENT_DATA",
        reason=reason,
        training_rows=training_rows,
        features_used=features_used or [],
        contamination=None,
        random_state=None,
        operating_state_method=operating_state_method,
        active_rows=active_rows,
        inactive_rows=inactive_rows,
        active_days_represented=active_days_represented,
        result_frame=None,
        debug_traceback=debug_traceback,
    )


def _count_qualifying_active_days(timestamps: pd.Series, min_rows_per_day: int) -> int:
    """Counts distinct UTC calendar dates represented among `timestamps`
    that individually contributed at least `min_rows_per_day` rows. See
    MIN_ACTIVE_ROWS_PER_DAY's rationale above for why a day must clear
    this bar to "count" toward day-diversity."""
    if len(timestamps) == 0:
        return 0
    dates = pd.to_datetime(timestamps, unit="s", utc=True).dt.date
    counts = dates.value_counts()
    return int((counts >= min_rows_per_day).sum())


# ---------------------------------------------------------------------------
# Per-PZEM entry point
# ---------------------------------------------------------------------------

def detect_anomalies_for_meter(
    pzem_number: int,
    settings: Optional[Settings] = None,
    preprocess_result: Optional[PreprocessResult] = None,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    min_active_training_rows: int = MIN_ACTIVE_TRAINING_ROWS,
    min_active_training_days: int = MIN_ACTIVE_TRAINING_DAYS,
    min_active_rows_per_day: int = MIN_ACTIVE_ROWS_PER_DAY,
    state_persistence_samples: int = DEFAULT_STATE_PERSISTENCE_SAMPLES,
    min_state_separation_ratio: float = MIN_STATE_SEPARATION_RATIO,
    min_state_component_weight: float = MIN_STATE_COMPONENT_WEIGHT,
    fallback_idle_percentile: float = FALLBACK_IDLE_PERCENTILE,
) -> AnomalyDetectionResult:
    """Runs Stage 3 for one PZEM: detects operating state, trains (or
    declines to train) an IsolationForest on ACTIVE-state history only,
    and scores ACTIVE rows.

    If preprocess_result is omitted, this calls
    ai.preprocessing.preprocess_meter() itself (the normal path — which
    in turn hits Stage 1's cache/Firebase). Tests pass preprocess_result
    directly to exercise the anomaly-detection logic without any Firebase
    or even Stage 1/2-fetch dependency at all.
    """
    settings = settings or get_settings()
    if preprocess_result is None:
        preprocess_result = preprocessing.preprocess_meter(pzem_number, settings=settings)

    if preprocess_result.status != "READY" or preprocess_result.feature_frame is None:
        return _insufficient_result(
            pzem_number,
            reason=(
                f"Stage 2 preprocessing status is {preprocess_result.status} "
                f"for this meter, so there is no feature frame to train on. "
                f"Stage 2 reason: {preprocess_result.reason}"
            ),
        )

    df = preprocess_result.feature_frame

    # --- Operating-state detection (electrical behavior only) ---
    state_result = detect_operating_state(
        df,
        persistence_samples=state_persistence_samples,
        min_separation_ratio=min_state_separation_ratio,
        min_component_weight=min_state_component_weight,
        fallback_idle_percentile=fallback_idle_percentile,
    )
    operating_state = state_result.labels  # aligned to df's row order

    features = _select_features(df)
    if not features:
        return _insufficient_result(
            pzem_number,
            reason="None of the candidate ML features were available in the Stage 2 output for this meter.",
            operating_state_method=state_result.method,
            active_rows=state_result.active_rows,
            inactive_rows=state_result.inactive_rows,
        )

    work = df[["timestamp", *features]].copy()
    work["operating_state"] = operating_state

    active_mask = work["operating_state"].to_numpy() == "ACTIVE"
    active_work = work[active_mask].reset_index(drop=True)

    # Rolling-window warm-up rows (the first few rows of any meter, where
    # min_periods hasn't been met yet) are NaN in the rolling columns even
    # for an otherwise READY meter — drop them from TRAINING rather than
    # imputing anything. Also defensively drop non-finite values (+/-inf)
    # even though Stage 2's cleaning is supposed to prevent them reaching
    # here: Stage 3 must not silently train on garbage just because an
    # upstream guarantee was violated. This is applied only to the ACTIVE
    # subset — INACTIVE rows never contribute to training regardless.
    before_na_drop = len(active_work)
    active_work = active_work.dropna(subset=features).reset_index(drop=True)
    if not active_work.empty:
        finite_mask = np.isfinite(active_work[features].to_numpy(dtype="float64")).all(axis=1)
        active_work = active_work[finite_mask].reset_index(drop=True)
    dropped_for_na_or_inf = before_na_drop - len(active_work)

    training_rows = len(active_work)
    active_days_represented = _count_qualifying_active_days(
        active_work["timestamp"], min_active_rows_per_day
    )

    if training_rows < min_active_training_rows or active_days_represented < min_active_training_days:
        reason = (
            f"Only {training_rows} ACTIVE row(s) with every selected feature "
            f"present and finite are available (dropped {dropped_for_na_or_inf} "
            f"row(s) to rolling-window warm-up NaNs/non-finite values out of "
            f"{before_na_drop} ACTIVE candidate rows; operating-state detection "
            f"({state_result.method}) found {state_result.active_rows} ACTIVE and "
            f"{state_result.inactive_rows} INACTIVE row(s) total for this meter). "
            f"At least {min_active_training_rows} ACTIVE rows spanning at least "
            f"{min_active_training_days} distinct day(s) with >= "
            f"{min_active_rows_per_day} ACTIVE rows each are required before an "
            f"IsolationForest is trained for this meter; only "
            f"{active_days_represented} qualifying day(s) were found."
        )
        return _insufficient_result(
            pzem_number,
            reason=reason,
            training_rows=training_rows,
            features_used=features,
            operating_state_method=state_result.method,
            active_rows=state_result.active_rows,
            inactive_rows=state_result.inactive_rows,
            active_days_represented=active_days_represented,
        )

    X = active_work[features].copy()
    if "is_weekend" in X.columns:
        X["is_weekend"] = X["is_weekend"].astype(int)
    X_values = X.to_numpy(dtype="float64")

    try:
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        model.fit(X_values)
        # decision_function: scikit-learn's own convention — HIGHER means
        # MORE NORMAL, negative values generally indicate anomalies. This
        # is a relative, unbounded real number, not a probability. See the
        # module docstring / AnomalyDetectionResult.result_frame columns
        # doc for how this is surfaced.
        raw_scores = model.decision_function(X_values)
        predictions = model.predict(X_values)  # 1 = normal, -1 = anomaly
    except Exception as exc:  # noqa: BLE001 - one meter's model failing must
        # not take down the fleet report; see run_anomaly_detection_pipeline.
        tb = traceback.format_exc()
        logger.error(
            "IsolationForest training/scoring failed for PZEM %d (training_rows=%d):\n%s",
            pzem_number, training_rows, tb,
        )
        return _insufficient_result(
            pzem_number,
            reason=f"IsolationForest training/scoring failed: {exc}",
            training_rows=training_rows,
            features_used=features,
            debug_traceback=tb,
            operating_state_method=state_result.method,
            active_rows=state_result.active_rows,
            inactive_rows=state_result.inactive_rows,
            active_days_represented=active_days_represented,
        )

    # Normalized score, for a future dashboard: min-max scaled to [0, 1]
    # using ONLY this meter's own ACTIVE training scores (never
    # cross-meter, never including INACTIVE rows), then inverted so
    # 1.0 = most anomalous / 0.0 = most normal WITHIN THIS METER'S OWN
    # ACTIVE HISTORY — a more intuitive dashboard convention than raw
    # decision_function's "higher = more normal". This is a relative
    # ranking within one meter's ACTIVE training window, explicitly NOT
    # a probability of failure or of anything else.
    score_min = float(raw_scores.min())
    score_max = float(raw_scores.max())
    if score_max > score_min:
        scaled = (raw_scores - score_min) / (score_max - score_min)
    else:
        # Every row scored identically (e.g. perfectly constant input) —
        # there is no meaningful ranking to produce, so leave everything
        # in the middle of the scale rather than fabricating a spread.
        scaled = np.full_like(raw_scores, 0.5)
    anomaly_score_normalized = 1.0 - scaled

    anomaly_label = np.where(predictions == -1, "ANOMALY", "NORMAL")

    # PROVISIONAL severity ONLY. This is a coarse bucketing of
    # anomaly_score_normalized, included because Stage 4 needs a
    # placeholder column to build proper fault-severity logic on top of —
    # it is NOT fault diagnosis, and rows that aren't even flagged as
    # ANOMALY get "N/A" rather than a severity, since severity of a
    # non-anomaly is not a meaningful concept yet.
    provisional_bucket = np.where(
        anomaly_score_normalized >= 0.85, "HIGH_PROVISIONAL",
        np.where(anomaly_score_normalized >= 0.6, "MEDIUM_PROVISIONAL", "LOW_PROVISIONAL"),
    )
    anomaly_severity_provisional = np.where(anomaly_label == "ANOMALY", provisional_bucket, "N/A")

    scored = pd.DataFrame({
        "timestamp": active_work["timestamp"].to_numpy(),
        "anomaly_score": raw_scores,
        "anomaly_score_normalized": anomaly_score_normalized,
        "anomaly_label": anomaly_label,
        "anomaly_severity_provisional": anomaly_severity_provisional,
    })

    # Full result_frame: every row from the Stage 2 feature frame, in
    # original order, carrying its operating_state. Only rows that made
    # it into `scored` (ACTIVE + feature-complete) get real anomaly
    # values; everything else (INACTIVE, or ACTIVE-but-dropped-for-NaN/
    # inf) is explicitly NOT_SCORED rather than silently omitted or
    # given a fabricated score.
    result_frame = pd.DataFrame({
        "pzem_id": pzem_number,
        "timestamp": df["timestamp"].to_numpy(),
        "operating_state": operating_state,
    })
    result_frame = result_frame.merge(scored, on="timestamp", how="left")
    unscored = result_frame["anomaly_label"].isna()
    result_frame.loc[unscored, "anomaly_label"] = "NOT_SCORED"
    result_frame.loc[unscored, "anomaly_severity_provisional"] = "N/A"
    # anomaly_score / anomaly_score_normalized stay NaN for unscored rows
    # (float columns) — never fabricated.

    return AnomalyDetectionResult(
        pzem_number=pzem_number,
        model_status="READY",
        reason=None,
        training_rows=training_rows,
        features_used=features,
        contamination=contamination,
        random_state=random_state,
        operating_state_method=state_result.method,
        active_rows=state_result.active_rows,
        inactive_rows=state_result.inactive_rows,
        active_days_represented=active_days_represented,
        result_frame=result_frame,
    )


# ---------------------------------------------------------------------------
# Fleet-level entry point
# ---------------------------------------------------------------------------

def run_anomaly_detection_pipeline(
    settings: Optional[Settings] = None,
    preprocess_results: Optional[dict[int, PreprocessResult]] = None,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    min_active_training_rows: int = MIN_ACTIVE_TRAINING_ROWS,
    min_active_training_days: int = MIN_ACTIVE_TRAINING_DAYS,
    min_active_rows_per_day: int = MIN_ACTIVE_ROWS_PER_DAY,
    state_persistence_samples: int = DEFAULT_STATE_PERSISTENCE_SAMPLES,
) -> dict[int, AnomalyDetectionResult]:
    """Runs Stage 3 for every configured PZEM (1..settings.pzem_count)
    independently. One meter's model failing (bad data shape, a training
    exception, anything) must not stop the others from being reported —
    same fleet-resilience contract as Stage 2's run_preprocessing_pipeline.

    If preprocess_results is omitted, this calls
    ai.preprocessing.run_preprocessing_pipeline() itself. Meter numbers
    are never hard-coded — the loop always runs 1..settings.pzem_count,
    so whichever PZEMs actually have data (which may change as the real
    classroom deployment accumulates history) are picked up automatically.
    """
    settings = settings or get_settings()
    if preprocess_results is None:
        preprocess_results = preprocessing.run_preprocessing_pipeline(settings=settings)

    results: dict[int, AnomalyDetectionResult] = {}
    for n in range(1, settings.pzem_count + 1):
        try:
            pre = preprocess_results.get(n)
            if pre is None:
                results[n] = _insufficient_result(
                    n, reason="No Stage 2 preprocessing result was available for this meter."
                )
                continue
            results[n] = detect_anomalies_for_meter(
                n,
                settings=settings,
                preprocess_result=pre,
                contamination=contamination,
                random_state=random_state,
                n_estimators=n_estimators,
                min_active_training_rows=min_active_training_rows,
                min_active_training_days=min_active_training_days,
                min_active_rows_per_day=min_active_rows_per_day,
                state_persistence_samples=state_persistence_samples,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad: one
            # meter's unexpected failure must not take down the fleet report.
            tb = traceback.format_exc()
            logger.error("Unexpected error running anomaly detection for PZEM %d:\n%s", n, tb)
            results[n] = _insufficient_result(
                n,
                reason=f"Unexpected error during anomaly detection: {exc}",
                debug_traceback=tb,
            )
    return results


def summarize_fleet(results: dict[int, AnomalyDetectionResult]) -> FleetAnomalySummary:
    """Fleet-level counts for the report: how many PZEMs were analyzed,
    how many couldn't be modeled yet, and — among the ones that could —
    how many are showing an anomaly RIGHT NOW (their most recent SCORED
    row) versus normal right now. A meter whose most recent row happens
    to be INACTIVE (and therefore NOT_SCORED) looks at its most recent
    scored row instead, since "currently idle" is not itself anomalous
    or normal in the Stage 3 sense."""
    analyzed = len(results)
    insufficient = sum(1 for r in results.values() if r.model_status != "READY")
    anomalous_now = 0
    normal_now = 0
    for r in results.values():
        if r.model_status == "READY" and r.result_frame is not None and not r.result_frame.empty:
            scored = r.result_frame[r.result_frame["anomaly_label"] != "NOT_SCORED"]
            if scored.empty:
                continue
            latest_label = scored.iloc[-1]["anomaly_label"]
            if latest_label == "ANOMALY":
                anomalous_now += 1
            else:
                normal_now += 1
    return FleetAnomalySummary(
        analyzed=analyzed,
        insufficient_data=insufficient,
        anomalous_now=anomalous_now,
        normal_now=normal_now,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(results: dict[int, AnomalyDetectionResult]) -> str:
    """Renders the per-PZEM Stage 3 report: operating-state detection
    summary, training rows/days, status, and (for READY meters) the
    latest scored anomaly result — for every PZEM, in order, regardless
    of which ones actually trained a model."""
    lines = []
    for n in sorted(results):
        r = results[n]
        lines.append(f"PZEM {n}")
        lines.append(f"Operating-state method: {r.operating_state_method}")
        lines.append(f"Active rows / Inactive rows: {r.active_rows} / {r.inactive_rows}")
        lines.append(f"Active days represented (>= min rows/day): {r.active_days_represented}")
        lines.append(f"Training rows (ACTIVE, feature-complete): {r.training_rows}")
        lines.append(f"Status: {r.model_status}")
        if r.model_status == "READY" and r.result_frame is not None and not r.result_frame.empty:
            scored = r.result_frame[r.result_frame["anomaly_label"] != "NOT_SCORED"]
            lines.append(f"Features used: {', '.join(r.features_used)}")
            lines.append(f"Contamination: {r.contamination}")
            lines.append(f"Random state: {r.random_state}")
            if not scored.empty:
                latest = scored.iloc[-1]
                lines.append(f"Latest scored anomaly: {latest['anomaly_label']}")
                lines.append(f"Anomaly score (raw decision_function): {latest['anomaly_score']:.6f}")
                lines.append(f"Anomaly score (normalized, this meter only): {latest['anomaly_score_normalized']:.4f}")
                lines.append(f"Anomaly severity (provisional): {latest['anomaly_severity_provisional']}")
            else:
                lines.append("No scored rows yet (most recent data is INACTIVE).")
        else:
            lines.append(f"Reason: {r.reason}")
        if r.debug_traceback:
            lines.append(
                "  (!) An exception occurred while processing this meter — "
                "see print_debug_tracebacks() output / logs for the full traceback."
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def print_debug_tracebacks(results: dict[int, AnomalyDetectionResult]) -> None:
    any_failures = False
    for n in sorted(results):
        r = results[n]
        if r.debug_traceback:
            any_failures = True
            print(f"\n{'=' * 70}\nFull traceback for PZEM {n}\n{'=' * 70}")
            print(r.debug_traceback)
    if not any_failures:
        print("No exceptions were caught during this run.")
