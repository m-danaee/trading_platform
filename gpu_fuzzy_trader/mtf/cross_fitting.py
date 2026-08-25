"""Master Temporal Folds, Purged Embargo, and Out-of-Fold (OOF) Cross-Fitting.

Coordinates expanding temporal train/test splits shared identically across all
timeframes (HWC, MWC, LWC), strictly enforces forward-lookahead purge embargoes,
and aggregates out-of-fold validation scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Mapping, Sequence, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Deprecated default names are initialized from config below.  New callers
# must use purge_for_role() so a changed horizon cannot create a second source.


@dataclass(frozen=True, init=False)
class TemporalFold:
    """Dataclass defining a single master temporal cross-fitting fold.

    Attributes:
        fold_id: 1-based integer index of the fold.
        train_start: Starting timestamp of the training interval.
        train_end: Nominal ending timestamp of the training interval (before purging).
        test_start: Starting timestamp of the out-of-fold test interval.
        test_end: Ending timestamp of the out-of-fold test interval.
    Purge is deliberately not part of a fold.  A fold describes geometry only;
    the caller selects the role-specific purge when rows are retrieved.
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __init__(
        self,
        fold_id: int,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        is_seed: bool | None = None,
    ) -> None:
        """Create geometry-only fold metadata.

        ``is_seed`` is accepted only as a deprecated constructor compatibility
        argument.  It is not a dataclass field and is never used for role
        eligibility.  Older callers that read the property still receive the
        Fold 1 geometry alias.
        """
        object.__setattr__(self, "fold_id", int(fold_id))
        object.__setattr__(self, "train_start", train_start)
        object.__setattr__(self, "train_end", train_end)
        object.__setattr__(self, "test_start", test_start)
        object.__setattr__(self, "test_end", test_end)
        object.__setattr__(
            self,
            "_legacy_seed_override",
            None if is_seed is None else bool(is_seed),
        )

    @property
    def is_seed(self) -> bool:
        """Deprecated read-only alias for the Fold 1 geometry."""
        override = getattr(self, "_legacy_seed_override", None)
        if override is not None:
            return bool(override)
        return self.fold_id == 1

    def to_dict(self) -> dict[str, Any]:
        """Convert fold metadata to JSON-serializable dictionary."""
        return {
            "fold_id": int(self.fold_id),
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }

    def get_train_slice(
        self,
        df: pd.DataFrame,
        purge_minutes: int | None = None,
        datetime_col: str = "datetime",
        *,
        role: str | None = None,
    ) -> pd.DataFrame:
        """Extract purged training data for this fold from a DataFrame."""
        if role is not None:
            p_min = _purge_minutes_for_role(role)
        elif purge_minutes is not None:
            p_min = _validate_purge_minutes(purge_minutes)
        else:
            # Keep the historical no-argument call working while new callers
            # should pass role explicitly.
            p_min = _purge_minutes_for_role("hwc")
        dt_series = _get_datetime_series(df, datetime_col)
        raw_train = df.loc[(dt_series >= self.train_start) & (dt_series < self.test_start)].copy()
        return apply_purge_embargo(
            train_df=raw_train,
            pred_start_dt=self.test_start,
            purge_minutes=p_min,
            datetime_col=datetime_col,
        )

    def get_test_slice(
        self,
        df: pd.DataFrame,
        datetime_col: str = "datetime",
        inclusive_end: bool = False,
    ) -> pd.DataFrame:
        """Extract out-of-fold test data for this fold from a DataFrame."""
        dt_series = _get_datetime_series(df, datetime_col)
        if inclusive_end:
            mask = (dt_series >= self.test_start) & (dt_series <= self.test_end)
        else:
            mask = (dt_series >= self.test_start) & (dt_series < self.test_end)
        return df.loc[mask].copy()


@dataclass(frozen=True)
class FoldExposure:
    """Rows and time exposure represented by one fold side.

    ``rows`` and ``duration_bars`` refer to the side used for the exposure
    (normally the OOF/test side).  The optional train/test fields make the
    same record useful for audit manifests without introducing another
    representation of fold geometry.
    """

    rows: int
    duration_bars: float
    per_symbol_rows: Mapping[str, int] = field(default_factory=dict)
    train_rows: int | None = None
    test_rows: int | None = None
    train_duration_bars: float | None = None
    test_duration_bars: float | None = None

    @property
    def effective_rows(self) -> int:
        """Return the row count used by count-gate scaling."""
        return int(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": int(self.rows),
            "duration_bars": float(self.duration_bars),
            "per_symbol_rows": {
                str(symbol): int(rows)
                for symbol, rows in self.per_symbol_rows.items()
            },
            "train_rows": (
                None if self.train_rows is None else int(self.train_rows)
            ),
            "test_rows": None if self.test_rows is None else int(self.test_rows),
            "train_duration_bars": (
                None
                if self.train_duration_bars is None
                else float(self.train_duration_bars)
            ),
            "test_duration_bars": (
                None
                if self.test_duration_bars is None
                else float(self.test_duration_bars)
            ),
        }


@dataclass(frozen=True)
class FoldEligibility:
    """Result of applying the fail-closed fold eligibility checks."""

    eligible: bool
    reason: str = ""
    failed_checks: tuple[str, ...] = ()
    symbol_coverage: float = 0.0
    symbols_expected: tuple[str, ...] = ()
    symbols_covered: tuple[str, ...] = ()

    @property
    def is_eligible(self) -> bool:
        """Readable alias used by callers that prefer predicate wording."""
        return bool(self.eligible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": bool(self.eligible),
            "reason": str(self.reason),
            "failed_checks": list(self.failed_checks),
            "symbol_coverage": float(self.symbol_coverage),
            "symbols_expected": list(self.symbols_expected),
            "symbols_covered": list(self.symbols_covered),
        }


@dataclass(frozen=True)
class FoldContext:
    """Auditable row, duration, symbol, and eligibility context for a fold."""

    train_rows: int
    test_rows: int
    per_symbol_rows: Mapping[str, Mapping[str, int]]
    train_duration: float
    test_duration: float
    symbols_available: tuple[str, ...]
    eligible: bool
    reason: str = ""
    per_symbol_train_rows: Mapping[str, int] = field(default_factory=dict)
    per_symbol_test_rows: Mapping[str, int] = field(default_factory=dict)
    symbols_expected: tuple[str, ...] = ()
    symbols_covered: tuple[str, ...] = ()
    symbol_coverage: float = 0.0
    train_duration_delta: pd.Timedelta = pd.Timedelta(0)
    test_duration_delta: pd.Timedelta = pd.Timedelta(0)
    exposure: FoldExposure | None = None
    eligibility: FoldEligibility | None = None

    @property
    def train_duration_bars(self) -> float:
        """Explicit name for the numeric train duration."""
        return float(self.train_duration)

    @property
    def test_duration_bars(self) -> float:
        """Explicit name for the numeric test duration."""
        return float(self.test_duration)

    @property
    def available_symbol_count(self) -> int:
        return len(self.symbols_available)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe audit record for this context."""
        eligibility = self.eligibility or FoldEligibility(
            eligible=bool(self.eligible),
            reason=str(self.reason),
            symbol_coverage=float(self.symbol_coverage),
            symbols_expected=tuple(self.symbols_expected),
            symbols_covered=tuple(self.symbols_covered),
        )
        return {
            "train_rows": int(self.train_rows),
            "test_rows": int(self.test_rows),
            "per_symbol_rows": {
                str(symbol): {
                    str(name): int(value) for name, value in counts.items()
                }
                for symbol, counts in self.per_symbol_rows.items()
            },
            "per_symbol_train_rows": {
                str(symbol): int(rows)
                for symbol, rows in self.per_symbol_train_rows.items()
            },
            "per_symbol_test_rows": {
                str(symbol): int(rows)
                for symbol, rows in self.per_symbol_test_rows.items()
            },
            "train_duration": float(self.train_duration),
            "test_duration": float(self.test_duration),
            "train_duration_bars": float(self.train_duration),
            "test_duration_bars": float(self.test_duration),
            "train_duration_delta": self.train_duration_delta.isoformat(),
            "test_duration_delta": self.test_duration_delta.isoformat(),
            "symbols_available": list(self.symbols_available),
            "symbols_expected": list(self.symbols_expected),
            "symbols_covered": list(self.symbols_covered),
            "symbol_coverage": float(self.symbol_coverage),
            "eligible": bool(self.eligible),
            "reason": str(self.reason),
            "eligibility": eligibility.to_dict(),
            "exposure": (
                None if self.exposure is None else self.exposure.to_dict()
            ),
        }


def _get_datetime_series(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.Series:
    """Extract and validate datetime Series from column or DatetimeIndex."""
    if df.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")

    if datetime_col in df.columns:
        dt = df[datetime_col]
        parsed = pd.to_datetime(dt, errors="raise", utc=True)
        return parsed.dt.tz_localize(None)
    if isinstance(df.index, pd.DatetimeIndex):
        parsed_index = pd.to_datetime(df.index, errors="raise", utc=True).tz_localize(None)
        return pd.Series(parsed_index, index=df.index)
    raise ValueError(
        f"Cannot find datetime column '{datetime_col}' or DatetimeIndex in DataFrame."
    )


def _config_value(name: str, default: Any) -> Any:
    """Read a folding default lazily to avoid making config a module dependency."""
    try:
        from gpu_fuzzy_trader import config as _cfg

        return getattr(_cfg, name, default)
    except Exception:  # pragma: no cover - defensive import fallback
        return default


def _validate_purge_minutes(value: Any) -> int:
    """Validate an explicit purge duration without storing it on a fold."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"purge_minutes must be a non-negative integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"purge_minutes must be non-negative, got {value!r}")
    return parsed


def _configured_default_purge(role: str) -> int:
    """Resolve a deprecated default name from the canonical config helper."""
    configured_helper = _config_value("purge_for_role", None)
    if not callable(configured_helper):
        raise RuntimeError("gpu_fuzzy_trader.config.purge_for_role is required")
    return _validate_purge_minutes(configured_helper(role))


# Deprecated compatibility aliases.  These values are derived from config and
# are kept only for callers that imported the old names.
DEFAULT_HWC_PURGE_MINUTES: int = _configured_default_purge("hwc")
DEFAULT_MWC_PURGE_MINUTES: int = _configured_default_purge("mwc")
DEFAULT_LWC_PURGE_MINUTES: int = _configured_default_purge("lwc")


_MTF_ROLES = frozenset({"hwc", "mwc", "lwc"})


def _normalize_role(role: str) -> str:
    """Normalize and validate an MTF role name."""
    normalized = str(role).strip().lower()
    if normalized not in _MTF_ROLES:
        raise ValueError(f"Unknown MTF role {role!r}; expected hwc, mwc, or lwc")
    return normalized


def eligible_for_role(fold: TemporalFold, role: str) -> bool:
    """Return whether ``fold`` may produce OOF data for an MTF role.

    HWC has no upstream layer, so every geometry fold is usable.  MWC and LWC
    consume upstream OOF evidence and therefore skip Fold 1, which has no
    preceding HWC OOF history.  LWC follows the same rule for now because its
    downstream evidence is also unavailable on the seed fold.
    """
    if not isinstance(fold, TemporalFold):
        raise TypeError("eligible_for_role expects a TemporalFold")
    normalized = _normalize_role(role)
    fold_id = int(fold.fold_id)
    if normalized == "hwc":
        return fold_id >= 1
    return fold_id > 1


def _purge_minutes_for_role(role: str) -> int:
    """Resolve a role purge from the single-source config helper."""
    normalized = _normalize_role(role)

    configured_helper = _config_value("purge_for_role", None)
    if callable(configured_helper):
        return _validate_purge_minutes(configured_helper(normalized))

    for configured_name in (
        f"{normalized.upper()}_PURGE",
        f"MTF_{normalized.upper()}_PURGE_MINUTES",
    ):
        configured = _config_value(configured_name, None)
        if configured is not None:
            return _validate_purge_minutes(configured)

    fallback = {
        "hwc": DEFAULT_HWC_PURGE_MINUTES,
        "mwc": DEFAULT_MWC_PURGE_MINUTES,
        "lwc": DEFAULT_LWC_PURGE_MINUTES,
    }
    return fallback[normalized]


def _fold_thresholds(
    *,
    min_effective_rows: int | None,
    min_rows_per_symbol: int | None,
    min_duration_bars: float | None,
    min_symbol_coverage: float | None,
) -> tuple[int, int, float, float]:
    """Resolve and validate fold eligibility thresholds."""
    effective_rows = int(
        _config_value("FOLD_MIN_EFFECTIVE_ROWS", 5)
        if min_effective_rows is None
        else min_effective_rows
    )
    effective_rows = max(
        effective_rows,
        int(_config_value("FOLD_ABSOLUTE_MIN_TRADES", 5)),
    )
    rows_per_symbol = int(
        _config_value("FOLD_MIN_ROWS_PER_SYMBOL", 5)
        if min_rows_per_symbol is None
        else min_rows_per_symbol
    )
    duration_bars = float(
        _config_value("FOLD_MIN_DURATION_BARS", 5)
        if min_duration_bars is None
        else min_duration_bars
    )
    symbol_coverage = float(
        _config_value("FOLD_MIN_SYMBOL_COVERAGE", 1.0)
        if min_symbol_coverage is None
        else min_symbol_coverage
    )

    if effective_rows < 1:
        raise ValueError("min_effective_rows must be positive")
    if rows_per_symbol < 1:
        raise ValueError("min_rows_per_symbol must be positive")
    if duration_bars < 0:
        raise ValueError("min_duration_bars must be non-negative")
    if not 0.0 < symbol_coverage <= 1.0:
        raise ValueError("min_symbol_coverage must be in (0, 1]")
    return effective_rows, rows_per_symbol, duration_bars, symbol_coverage


def _infer_bar_delta(dt_series: pd.Series) -> pd.Timedelta:
    """Infer the stable bar cadence from unique timestamps."""
    unique = pd.DatetimeIndex(pd.Series(dt_series).dropna().drop_duplicates().sort_values())
    if len(unique) < 2:
        return pd.Timedelta(minutes=1)
    deltas = unique.to_series().diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return pd.Timedelta(minutes=1)
    return pd.Timedelta(deltas.median())


def _duration_in_bars(start: pd.Timestamp, end: pd.Timestamp, bar_delta: pd.Timedelta) -> float:
    """Convert a temporal interval to a cadence-normalized bar duration."""
    if end <= start:
        return 0.0
    return float((end - start).total_seconds() / bar_delta.total_seconds())


def _symbol_series(
    df: pd.DataFrame,
    *,
    symbol_col: str,
) -> pd.Series:
    """Return stable string symbols, including a single-symbol fallback."""
    if symbol_col in df.columns:
        values = df[symbol_col].astype("string").fillna("__UNKNOWN__")
        return values.astype(str)
    return pd.Series("__ALL_SYMBOLS__", index=df.index, dtype="object")


def _interval_mask(
    dt_series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    inclusive_end: bool = False,
) -> pd.Series:
    """Build a half-open interval mask, optionally including the final boundary."""
    if inclusive_end:
        return (dt_series >= start) & (dt_series <= end)
    return (dt_series >= start) & (dt_series < end)


def _build_temporal_folds(
    dt_series: pd.Series,
    n_folds: int,
) -> list[TemporalFold]:
    """Build equal-time expanding folds without applying eligibility rules."""
    t_min = dt_series.min()
    t_max = dt_series.max()
    if pd.isna(t_min) or pd.isna(t_max) or t_min >= t_max:
        raise ValueError("DataFrame datetime span must have at least 2 distinct timestamps.")

    n_blocks = n_folds + 1
    step_delta = (t_max - t_min) / n_blocks
    boundaries: list[pd.Timestamp] = [t_min + i * step_delta for i in range(n_blocks)]
    boundaries.append(t_max)
    return [
        TemporalFold(
            fold_id=k,
            train_start=boundaries[0],
            train_end=boundaries[k],
            test_start=boundaries[k],
            test_end=boundaries[k + 1],
        )
        for k in range(1, n_folds + 1)
    ]


def assess_fold_eligibility(
    fold_or_df: TemporalFold | pd.DataFrame | None = None,
    df_or_fold: pd.DataFrame | TemporalFold | None = None,
    *,
    fold: TemporalFold | None = None,
    df: pd.DataFrame | None = None,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    min_effective_rows: int | None = None,
    min_rows_per_symbol: int | None = None,
    min_duration_bars: float | None = None,
    min_symbol_coverage: float | None = None,
    expected_symbols: Sequence[str] | None = None,
    inclusive_test_end: bool | None = None,
) -> FoldContext:
    """Assess one fold and return its complete audit context.

    Both ``assess_fold_eligibility(fold, df)`` and the intuitive
    ``assess_fold_eligibility(df, fold)`` order are accepted.  Durations are
    cadence-normalized bar durations, not row counts, so missing bars cannot
    make a short calendar interval appear healthy.
    """
    if fold is not None or df is not None:
        if fold is None or df is None or fold_or_df is not None or df_or_fold is not None:
            raise TypeError("provide either positional fold/df or keyword fold/df")
        fold_obj, frame = fold, df
    elif isinstance(fold_or_df, TemporalFold) and isinstance(df_or_fold, pd.DataFrame):
        fold_obj, frame = fold_or_df, df_or_fold
    elif isinstance(fold_or_df, pd.DataFrame) and isinstance(df_or_fold, TemporalFold):
        frame, fold_obj = fold_or_df, df_or_fold
    else:
        raise TypeError("assess_fold_eligibility expects a TemporalFold and a DataFrame")

    if frame.empty:
        raise ValueError("Cannot assess a fold from an empty DataFrame.")

    effective_rows, rows_per_symbol, duration_floor, coverage_floor = _fold_thresholds(
        min_effective_rows=min_effective_rows,
        min_rows_per_symbol=min_rows_per_symbol,
        min_duration_bars=min_duration_bars,
        min_symbol_coverage=min_symbol_coverage,
    )
    dt_series = _get_datetime_series(frame, datetime_col)
    symbols = _symbol_series(frame, symbol_col=symbol_col)
    all_symbols = tuple(
        sorted(
            str(symbol)
            for symbol in (
                expected_symbols
                if expected_symbols is not None
                else symbols.drop_duplicates().tolist()
            )
        )
    )
    if not all_symbols:
        all_symbols = ("__ALL_SYMBOLS__",)

    if inclusive_test_end is None:
        inclusive_test_end = bool(fold_obj.test_end >= dt_series.max())
    train_mask = _interval_mask(dt_series, fold_obj.train_start, fold_obj.train_end)
    test_mask = _interval_mask(
        dt_series,
        fold_obj.test_start,
        fold_obj.test_end,
        inclusive_end=bool(inclusive_test_end),
    )
    train_rows = int(train_mask.sum())
    test_rows = int(test_mask.sum())
    train_duration_delta = fold_obj.train_end - fold_obj.train_start
    test_duration_delta = fold_obj.test_end - fold_obj.test_start
    bar_delta = _infer_bar_delta(dt_series)
    train_duration = _duration_in_bars(
        fold_obj.train_start, fold_obj.train_end, bar_delta
    )
    test_duration = _duration_in_bars(
        fold_obj.test_start, fold_obj.test_end, bar_delta
    )

    per_symbol_train_rows = {
        symbol: int((train_mask & (symbols == symbol)).sum())
        for symbol in all_symbols
    }
    per_symbol_test_rows = {
        symbol: int((test_mask & (symbols == symbol)).sum())
        for symbol in all_symbols
    }
    per_symbol_rows = {
        symbol: {
            "train": per_symbol_train_rows[symbol],
            "test": per_symbol_test_rows[symbol],
            "total": per_symbol_train_rows[symbol] + per_symbol_test_rows[symbol],
            "train_rows": per_symbol_train_rows[symbol],
            "test_rows": per_symbol_test_rows[symbol],
            "total_rows": per_symbol_train_rows[symbol] + per_symbol_test_rows[symbol],
        }
        for symbol in all_symbols
    }
    symbols_available = tuple(
        symbol
        for symbol in all_symbols
        if per_symbol_train_rows[symbol] > 0 and per_symbol_test_rows[symbol] > 0
    )
    symbols_covered = tuple(
        symbol
        for symbol in all_symbols
        if per_symbol_train_rows[symbol] >= rows_per_symbol
        and per_symbol_test_rows[symbol] >= rows_per_symbol
    )
    symbol_coverage = len(symbols_covered) / len(all_symbols)

    failed: list[str] = []
    if train_rows < effective_rows:
        failed.append(f"train_rows={train_rows} < {effective_rows}")
    if test_rows < effective_rows:
        failed.append(f"test_rows={test_rows} < {effective_rows}")
    if train_duration < duration_floor:
        failed.append(
            f"train_duration_bars={train_duration:.6g} < {duration_floor:g}"
        )
    if test_duration < duration_floor:
        failed.append(
            f"test_duration_bars={test_duration:.6g} < {duration_floor:g}"
        )
    missing_symbols = [symbol for symbol in all_symbols if symbol not in symbols_covered]
    if missing_symbols:
        failed.append(
            "per_symbol_rows below minimum for " + ", ".join(missing_symbols)
        )
    if symbol_coverage < coverage_floor:
        failed.append(
            f"symbol_coverage={symbol_coverage:.6g} < {coverage_floor:g}"
        )

    reason = "; ".join(failed)
    eligibility = FoldEligibility(
        eligible=not failed,
        reason=reason,
        failed_checks=tuple(failed),
        symbol_coverage=symbol_coverage,
        symbols_expected=all_symbols,
        symbols_covered=symbols_covered,
    )
    exposure = FoldExposure(
        rows=test_rows,
        duration_bars=test_duration,
        per_symbol_rows=per_symbol_test_rows,
        train_rows=train_rows,
        test_rows=test_rows,
        train_duration_bars=train_duration,
        test_duration_bars=test_duration,
    )
    return FoldContext(
        train_rows=train_rows,
        test_rows=test_rows,
        per_symbol_rows=per_symbol_rows,
        train_duration=train_duration,
        test_duration=test_duration,
        symbols_available=symbols_available,
        eligible=eligibility.eligible,
        reason=eligibility.reason,
        per_symbol_train_rows=per_symbol_train_rows,
        per_symbol_test_rows=per_symbol_test_rows,
        symbols_expected=all_symbols,
        symbols_covered=symbols_covered,
        symbol_coverage=symbol_coverage,
        train_duration_delta=train_duration_delta,
        test_duration_delta=test_duration_delta,
        exposure=exposure,
        eligibility=eligibility,
    )


def build_fold_contexts(
    df_or_folds: pd.DataFrame | Sequence[TemporalFold] | None = None,
    folds_or_df: Sequence[TemporalFold] | pd.DataFrame | None = None,
    *,
    df: pd.DataFrame | None = None,
    folds: Sequence[TemporalFold] | None = None,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    min_effective_rows: int | None = None,
    min_rows_per_symbol: int | None = None,
    min_duration_bars: float | None = None,
    min_symbol_coverage: float | None = None,
    expected_symbols: Sequence[str] | None = None,
) -> list[FoldContext]:
    """Build audit contexts for all folds without filtering them."""
    if df is not None or folds is not None:
        if (
            df is None
            or folds is None
            or df_or_folds is not None
            or folds_or_df is not None
        ):
            raise TypeError("provide either positional df/folds or keyword df/folds")
        frame, fold_values = df, folds
    elif isinstance(df_or_folds, pd.DataFrame):
        frame, fold_values = df_or_folds, folds_or_df
    else:
        fold_values, frame = df_or_folds, folds_or_df
    if not isinstance(frame, pd.DataFrame) or fold_values is None:
        raise TypeError("build_fold_contexts expects a DataFrame and fold sequence")
    fold_list = list(fold_values)
    if not all(isinstance(fold, TemporalFold) for fold in fold_list):
        raise TypeError("build_fold_contexts received a non-TemporalFold item")
    return [
        assess_fold_eligibility(
            fold,
            frame,
            datetime_col=datetime_col,
            symbol_col=symbol_col,
            min_effective_rows=min_effective_rows,
            min_rows_per_symbol=min_rows_per_symbol,
            min_duration_bars=min_duration_bars,
            min_symbol_coverage=min_symbol_coverage,
            expected_symbols=expected_symbols,
            inclusive_test_end=index == len(fold_list) - 1,
        )
        for index, fold in enumerate(fold_list)
    ]


def build_master_temporal_folds(
    df: pd.DataFrame,
    max_folds: int | None = None,
    embargo_minutes: int | None = None,
    datetime_col: str = "datetime",
    *,
    n_folds: int | None = None,
    min_folds: int | None = None,
    min_effective_rows: int | None = None,
    min_rows_per_symbol: int | None = None,
    min_duration_bars: float | None = None,
    min_symbol_coverage: float | None = None,
    symbol_col: str = "symbol",
    expected_symbols: Sequence[str] | None = None,
    adaptive: bool | None = None,
    return_contexts: bool = False,
) -> list[TemporalFold] | tuple[list[TemporalFold], list[FoldContext]]:
    """Build equal-time expanding folds, adaptively selecting an eligible K.

    New callers use ``max_folds`` and receive the fail-closed adaptive
    behaviour (K=max_folds, max_folds-1, ..., min_folds).  The historical
    ``n_folds=`` keyword follows the same adaptive path when used alone.  A
    call that supplies the deprecated zero ``embargo_minutes`` argument keeps
    fixed-geometry compatibility so old audit callers can inspect an
    ineligible geometry; pass ``adaptive=True`` to opt back into fail-closed
    selection for that signature.

    ``embargo_minutes`` is accepted as a deprecated compatibility argument.
    Purge is not stored on ``TemporalFold``; use ``get_train_slice`` or
    ``generate_oof_scores`` with an explicit role/purge.
    """
    if df.empty:
        raise ValueError("Cannot build temporal folds from an empty DataFrame.")
    if max_folds is not None and n_folds is not None and int(max_folds) != int(n_folds):
        raise ValueError("max_folds and n_folds must match when both are provided")
    if max_folds is None:
        max_value = (
            int(n_folds)
            if n_folds is not None
            else int(_config_value("MTF_MAX_FOLDS", 4))
        )
    else:
        max_value = int(max_folds)
    invalid_name = "n_folds" if n_folds is not None and max_folds is None else "max_folds"
    if max_value < 1:
        raise ValueError(f"{invalid_name} must be >= 1, got {max_value}")
    if embargo_minutes is not None:
        _validate_purge_minutes(embargo_minutes)

    legacy_n_folds = (
        n_folds is not None
        and max_folds is None
        and adaptive is None
        and embargo_minutes == 0
        and min_folds is None
        and min_effective_rows is None
        and min_rows_per_symbol is None
        and min_duration_bars is None
        and min_symbol_coverage is None
    )
    use_adaptive = (
        not legacy_n_folds
        if adaptive is None
        else bool(adaptive)
    )
    dt_series = _get_datetime_series(df, datetime_col)

    if not use_adaptive:
        folds = _build_temporal_folds(dt_series, max_value)
        return (folds, build_fold_contexts(
            df,
            folds,
            datetime_col=datetime_col,
            symbol_col=symbol_col,
            min_effective_rows=min_effective_rows,
            min_rows_per_symbol=min_rows_per_symbol,
            min_duration_bars=min_duration_bars,
            min_symbol_coverage=min_symbol_coverage,
            expected_symbols=expected_symbols,
        )) if return_contexts else folds

    minimum = int(
        _config_value("MTF_MIN_FOLDS", 2) if min_folds is None else min_folds
    )
    if minimum < 1:
        raise ValueError(f"min_folds must be >= 1, got {minimum}")
    if minimum > max_value:
        raise ValueError(
            f"min_folds ({minimum}) cannot exceed max_folds ({max_value})"
        )

    failures: list[str] = []
    for candidate in range(max_value, minimum - 1, -1):
        folds = _build_temporal_folds(dt_series, candidate)
        contexts = build_fold_contexts(
            df,
            folds,
            datetime_col=datetime_col,
            symbol_col=symbol_col,
            min_effective_rows=min_effective_rows,
            min_rows_per_symbol=min_rows_per_symbol,
            min_duration_bars=min_duration_bars,
            min_symbol_coverage=min_symbol_coverage,
            expected_symbols=expected_symbols,
        )
        if all(context.eligible for context in contexts):
            return (folds, contexts) if return_contexts else folds
        failed_reasons = [
            f"fold {index + 1}: {context.reason}"
            for index, context in enumerate(contexts)
            if not context.eligible
        ]
        failures.append(f"K={candidate} ({'; '.join(failed_reasons)})")

    detail = "; ".join(failures)
    raise ValueError(f"no eligible adaptive fold count: {detail}")


def apply_purge_embargo(
    train_df: pd.DataFrame,
    pred_start_dt: Union[pd.Timestamp, str, None] = None,
    purge_minutes: int = 0,
    datetime_col: str = "datetime",
    *,
    test_start: Union[pd.Timestamp, str, None] = None,
) -> pd.DataFrame:
    """Purge training rows whose forward label/trade outcome horizon extends into test_start.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set DataFrame.
    pred_start_dt : pd.Timestamp, str, or None
        Start timestamp of the prediction / out-of-fold evaluation window.
    purge_minutes : int, default 0
        Embargo / forward horizon gap in minutes to strictly purge.
    datetime_col : str, default "datetime"
        Name of datetime column.
    test_start : pd.Timestamp, str, or None, optional
        Keyword alias for pred_start_dt.

    Returns
    -------
    pd.DataFrame
        Filtered training DataFrame guaranteed to contain only bars with
        timestamp < (test_start - purge_minutes).
    """
    if train_df.empty or purge_minutes <= 0:
        return train_df.copy()

    start_dt = pred_start_dt if pred_start_dt is not None else test_start
    if start_dt is None:
        raise ValueError("Either pred_start_dt or test_start must be provided.")

    target_dt = pd.to_datetime(start_dt, errors="raise", utc=True).tz_localize(None)
    cutoff_dt = target_dt - pd.Timedelta(minutes=int(purge_minutes))

    if datetime_col in train_df.columns:
        dt_series = train_df[datetime_col]
        dt_series = pd.to_datetime(dt_series, errors="raise", utc=True).dt.tz_localize(None)
        # A label whose forward horizon closes exactly at the prediction start
        # still touches the prediction interval and must be purged.
        mask = dt_series < cutoff_dt
        return train_df.loc[mask].copy()
    elif isinstance(train_df.index, pd.DatetimeIndex):
        index = pd.to_datetime(train_df.index, errors="raise", utc=True).tz_localize(None)
        mask = index < cutoff_dt
        return train_df.loc[mask].copy()
    else:
        raise ValueError(
            f"Cannot find datetime column '{datetime_col}' or DatetimeIndex in train_df."
        )


def generate_oof_scores(
    df: pd.DataFrame,
    folds: Sequence[TemporalFold],
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame, TemporalFold], Any],
    purge_minutes: int | None = None,
    datetime_col: str = "datetime",
    exclude_seed: bool | None = None,
    role: str | None = None,
) -> pd.DataFrame:
    """Coordinate fitting on purged training folds and generating out-of-fold predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Full master dataset DataFrame containing features and timestamps.
    folds : Sequence[TemporalFold]
        Temporal folds built by ``build_master_temporal_folds``.
    fit_predict_fn : Callable[[pd.DataFrame, pd.DataFrame, TemporalFold], Any]
        Callback receiving (purged_train_df, test_df, fold) and returning
        predictions (as DataFrame, Series, ndarray, or dict) aligned to test_df.
    purge_minutes : int or None, default None
        Override purge minutes for training set. If None, the configured purge
        for ``role`` (or HWC when no role is supplied) is used.
    datetime_col : str, default "datetime"
        Name of the datetime column.
    exclude_seed : bool or None, default None
        Deprecated compatibility alias.  Role-aware calls use
        ``eligible_for_role``; role-less calls retain the historical default of
        excluding Fold 1.
    role : {"hwc", "mwc", "lwc"} or None, optional
        Role used to resolve purge and seed-fold eligibility.

    Returns
    -------
    pd.DataFrame
        Combined out-of-fold prediction DataFrame with ``fold_id`` and
        prediction fields.  Role-less compatibility calls also include the
        deprecated ``is_seed`` output alias.
    """
    normalized_role = None if role is None else _normalize_role(role)
    if df.empty or not folds:
        return pd.DataFrame()

    dt_series = _get_datetime_series(df, datetime_col)
    n_folds = len(folds)
    results: list[pd.DataFrame] = []
    # A role is authoritative.  The old boolean can still request an
    # additional Fold 1 exclusion, but it can never make an ineligible role
    # eligible again.
    eligibility_role = (
        normalized_role
        if normalized_role is not None
        else "hwc"
        if exclude_seed is False
        else "mwc"
    )
    p_min = (
        _purge_minutes_for_role(normalized_role or "hwc")
        if purge_minutes is None
        else _validate_purge_minutes(purge_minutes)
    )

    for i, fold in enumerate(folds):
        if not eligible_for_role(fold, eligibility_role):
            continue
        if normalized_role is not None and exclude_seed is True and fold.fold_id == 1:
            continue

        # 1. Train slice: [train_start, test_start)
        train_mask = (dt_series >= fold.train_start) & (dt_series < fold.test_start)
        raw_train = df.loc[train_mask].copy()
        purged_train = apply_purge_embargo(
            train_df=raw_train,
            pred_start_dt=fold.test_start,
            purge_minutes=p_min,
            datetime_col=datetime_col,
        )

        # 2. Test slice: [test_start, test_end) or [test_start, test_end] for the last fold
        is_last = (i == n_folds - 1)
        if is_last:
            test_mask = (dt_series >= fold.test_start) & (dt_series <= fold.test_end)
        else:
            test_mask = (dt_series >= fold.test_start) & (dt_series < fold.test_end)

        test_slice = df.loc[test_mask].copy()
        if test_slice.empty:
            continue

        # 3. Fit & Predict callback
        preds = fit_predict_fn(purged_train, test_slice, fold)
        pred_df = _format_fold_predictions(
            preds,
            test_slice,
            fold,
            datetime_col,
            include_legacy_seed_alias=normalized_role is None,
        )
        results.append(pred_df)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, axis=0, ignore_index=True)
    if datetime_col in combined.columns:
        combined = combined.sort_values(datetime_col).reset_index(drop=True)

    return combined


# Alias for generate_oof_scores
generate_oof_predictions = generate_oof_scores


def _format_fold_predictions(
    preds: Any,
    test_slice: pd.DataFrame,
    fold: TemporalFold,
    datetime_col: str,
    *,
    include_legacy_seed_alias: bool = False,
) -> pd.DataFrame:
    """Format callback prediction output into a standardized DataFrame."""
    if isinstance(preds, pd.DataFrame):
        out = preds.copy()
    elif isinstance(preds, pd.Series):
        col_name = preds.name if preds.name else "prediction"
        out = pd.DataFrame({col_name: preds.values}, index=test_slice.index)
    elif isinstance(preds, np.ndarray):
        if preds.ndim == 1:
            out = pd.DataFrame({"prediction": preds}, index=test_slice.index)
        else:
            cols = [f"pred_{j}" for j in range(preds.shape[1])]
            out = pd.DataFrame(preds, columns=cols, index=test_slice.index)
    elif isinstance(preds, dict):
        out = pd.DataFrame(preds, index=test_slice.index)
    else:
        out = pd.DataFrame({"prediction": preds}, index=test_slice.index)

    # Attach datetime and symbol if present in test_slice and not in out
    if datetime_col in test_slice.columns and datetime_col not in out.columns:
        out.insert(0, datetime_col, test_slice[datetime_col].values)
    if "symbol" in test_slice.columns and "symbol" not in out.columns:
        out.insert(1, "symbol", test_slice["symbol"].values)

    out["fold_id"] = fold.fold_id
    if include_legacy_seed_alias:
        out["is_seed"] = fold.fold_id == 1
    return out


class _FoldBoundaryRecord(dict[str, Any]):
    """Geometry mapping with a read-only compatibility lookup for old code.

    The deprecated seed marker is intentionally not stored in the mapping, so
    manifests and JSON payloads contain geometry only.  Direct indexing by an
    older in-memory caller remains supported for one compatibility cycle.
    """

    def __init__(self, data: dict[str, Any], fold: TemporalFold) -> None:
        super().__init__(data)
        self._fold = fold

    def __getitem__(self, key: str) -> Any:
        if key == "is_seed":
            return self._fold.fold_id == 1
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "is_seed":
            return self._fold.fold_id == 1
        return super().get(key, default)


def export_fold_boundaries(
    folds: Sequence[TemporalFold],
    contexts: Sequence[FoldContext] | pd.DataFrame | None = None,
    *,
    df: pd.DataFrame | None = None,
    datetime_col: str = "datetime",
    symbol_col: str = "symbol",
    min_effective_rows: int | None = None,
    min_rows_per_symbol: int | None = None,
    min_duration_bars: float | None = None,
    min_symbol_coverage: float | None = None,
) -> list[dict[str, Any]]:
    """Export geometry and optional per-fold eligibility audit fields.

    Existing callers that pass only ``folds`` receive the same geometry fields
    as before, except that the deprecated ``purge_minutes`` field is absent.
    Pass ``contexts`` or ``df`` to include row, duration, symbol, and eligibility
    evidence in the manifest.
    """
    fold_list = list(folds)
    resolved_contexts: Sequence[FoldContext] | None
    if isinstance(contexts, pd.DataFrame):
        resolved_contexts = build_fold_contexts(
            contexts,
            fold_list,
            datetime_col=datetime_col,
            symbol_col=symbol_col,
            min_effective_rows=min_effective_rows,
            min_rows_per_symbol=min_rows_per_symbol,
            min_duration_bars=min_duration_bars,
            min_symbol_coverage=min_symbol_coverage,
        )
    elif contexts is None and df is not None:
        resolved_contexts = build_fold_contexts(
            df,
            fold_list,
            datetime_col=datetime_col,
            symbol_col=symbol_col,
            min_effective_rows=min_effective_rows,
            min_rows_per_symbol=min_rows_per_symbol,
            min_duration_bars=min_duration_bars,
            min_symbol_coverage=min_symbol_coverage,
        )
    else:
        resolved_contexts = contexts

    if resolved_contexts is not None and len(resolved_contexts) != len(fold_list):
        raise ValueError("contexts length must match folds length")

    exported: list[dict[str, Any]] = []
    for index, fold in enumerate(fold_list):
        record = fold.to_dict()
        if resolved_contexts is not None:
            record.update(resolved_contexts[index].to_dict())
        exported.append(_FoldBoundaryRecord(record, fold))
    return exported


def export_fold_contexts(
    folds: Sequence[TemporalFold],
    contexts: Sequence[FoldContext],
) -> list[dict[str, Any]]:
    """Explicit manifest helper for callers that already built contexts."""
    return export_fold_boundaries(folds, contexts)


def validate_master_temporal_folds(folds: Sequence[TemporalFold]) -> bool:
    """Validate that temporal folds are strictly monotonic and expanding.

    Checks:
    1. Fold IDs are strictly sequential starting at 1.
    2. Train start is identical across all folds.
    3. Train end == test start for every fold.
    4. Test intervals are contiguous (test_end_{k} == test_start_{k+1}).
    """
    if not folds:
        return False

    first = folds[0]
    if first.fold_id != 1:
        return False

    base_train_start = first.train_start

    for i, fold in enumerate(folds):
        expected_id = i + 1
        if fold.fold_id != expected_id:
            return False
        if fold.train_start != base_train_start:
            return False
        if fold.train_end != fold.test_start:
            return False
        if fold.test_start >= fold.test_end:
            return False
        # A legacy constructor override is checked only to avoid silently
        # accepting a corrupted object from an older caller.  Generated folds
        # have no override and validation is purely geometric.
        legacy_override = getattr(fold, "_legacy_seed_override", None)
        if (
            legacy_override is not None
            and bool(legacy_override) != (int(fold.fold_id) == 1)
        ):
            return False
        if i > 0:
            prev_fold = folds[i - 1]
            if prev_fold.test_end != fold.test_start:
                return False

    return True
