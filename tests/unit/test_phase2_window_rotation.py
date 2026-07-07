"""Tests for per-epoch train-window rotation (task-1)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.phases.phase2_island_scheduler import _derive_epoch_seed
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _resolve_sample_total_rows,
    _sample_df,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_multi_sym_df(
    n_rows_per_sym: int = 200,
    n_symbols: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a minimal multi-symbol DataFrame suitable for _sample_df tests."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for sym_idx in range(n_symbols):
        sym = f"SYM_{sym_idx}"
        data = {
            "datetime": pd.date_range(
                "2020-01-01", periods=n_rows_per_sym, freq="5min",
            ),
            "symbol": sym,
            "label_open_next": rng.uniform(100, 200, size=n_rows_per_sym),
            "label_close_288": rng.uniform(95, 210, size=n_rows_per_sym),
            "label_min_288": rng.uniform(90, 200, size=n_rows_per_sym),
            "label_max_288": rng.uniform(100, 220, size=n_rows_per_sym),
            "label_max_before_min": rng.integers(0, 2, size=n_rows_per_sym).astype(
                float,
            ),
            "_symbol_bar_index": np.arange(n_rows_per_sym),
        }
        for i in range(3):
            data[f"feat_{i}"] = rng.uniform(0, 1, size=n_rows_per_sym)
        parts.append(pd.DataFrame(data))
    return pd.concat(parts, ignore_index=True)


def _symbol_bar_min(df: pd.DataFrame) -> dict[str, int]:
    """Return per-symbol _symbol_bar_index.min() for a sampled DataFrame."""
    if df.empty:
        return {}
    return {
        sym: int(group["_symbol_bar_index"].min())
        for sym, group in df.groupby("symbol")
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeriveEpochSeed:
    """Deterministic seed derivation for per-epoch windows."""

    def test_deterministic(self):
        """Same (base_seed, epoch_idx) produces the same seed."""
        s1 = _derive_epoch_seed(12345, 0)
        s2 = _derive_epoch_seed(12345, 0)
        assert s1 == s2
        assert s1 is not None

    def test_epochs_differ(self):
        """Different epoch indices produce different seeds."""
        s0 = _derive_epoch_seed(12345, 0)
        s1 = _derive_epoch_seed(12345, 1)
        s7 = _derive_epoch_seed(12345, 7)
        assert s0 != s1
        assert s0 != s7
        assert s1 != s7

    def test_none_base(self):
        """None base seed returns None."""
        assert _derive_epoch_seed(None, 0) is None


class TestResolveSampleTotalRows:
    """Capping logic for per-epoch window rotation."""

    def test_no_cap_when_rotation_disabled(self, monkeypatch):
        """With PHASE2_PER_EPOCH_WINDOW_ROTATION=False, total_rows is unchanged."""
        monkeypatch.setattr(cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", False)
        df = _make_multi_sym_df(n_rows_per_sym=200, n_symbols=4)
        result = _resolve_sample_total_rows(df, 701_000)
        assert result == 701_000

    def test_caps_to_fit_safe_range(self, monkeypatch):
        """With rotation enabled, total_rows is capped with 1% margin."""
        monkeypatch.setattr(cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True)
        # 4 symbols, 200 rows each → safe_len = 200 (no forbidden ranges)
        # margin = max(1, 200//100) = 2, max_per_sym = 198
        # cap = min(701_000, 4 * 198) = 792
        df = _make_multi_sym_df(n_rows_per_sym=200, n_symbols=4)
        result = _resolve_sample_total_rows(df, 701_000)
        expected = 4 * (200 - max(1, 200 // 100))  # 4 * 198 = 792
        assert result == expected, f"Expected {expected}, got {result}"

    def test_cap_respects_forbidden_ranges(self, monkeypatch):
        """Forbidden ranges reduce safe_len, so the cap is tighter."""
        monkeypatch.setattr(cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True)
        # 4 symbols, 300 rows each, forbid bars 250-299 → safe_len = 250
        # margin = max(1, 250//100) = 2, max_per_sym = 248
        # cap = min(701_000, 4 * 248) = 992
        df = _make_multi_sym_df(n_rows_per_sym=300, n_symbols=4)
        forbidden = [(250, 299)]  # last 50 bars forbidden
        result = _resolve_sample_total_rows(df, 701_000, forbidden)
        expected = 4 * (250 - max(1, 250 // 100))  # 4 * 248 = 992
        assert result == expected, f"Expected {expected}, got {result}"

    def test_no_cap_when_request_fits(self, monkeypatch):
        """If total_rows already fits, the cap is a no-op."""
        monkeypatch.setattr(cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True)
        df = _make_multi_sym_df(n_rows_per_sym=500, n_symbols=2)
        # safe_len = 500, margin = 5, max_per_sym = 495
        # n_sym * max_per_sym = 2 * 495 = 990 > 800 → no cap
        result = _resolve_sample_total_rows(df, 800)
        assert result == 800  # unchanged


class TestSampleEpochRotation:
    """The sampled train windows differ across epochs."""

    def test_deterministic_per_epoch(self, monkeypatch):
        """Same (df, total_rows, seed) produces identical output."""
        monkeypatch.setattr(cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True)
        df = _make_multi_sym_df(n_rows_per_sym=500, n_symbols=4)
        seed = 12345
        sampled_a = _sample_df(df, 4 * 400, random_state=seed)
        sampled_b = _sample_df(df, 4 * 400, random_state=seed)
        assert sampled_a.equals(sampled_b)

    def test_start_bar_rotates_across_epochs(self, monkeypatch):
        """Different epoch seeds → different _symbol_bar_index.min() per sym.

        This is the core acceptance criterion: each epoch sees a different
        contiguous sub-window of the training data.
        """
        monkeypatch.setattr(cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True)
        # Use a larger DataFrame so there's room to rotate
        df = _make_multi_sym_df(n_rows_per_sym=1000, n_symbols=4)
        base_seed = 9999
        total_rows = 4 * 600  # request 600 rows/sym, safe_len=1000 → fits

        epoch_seeds = [_derive_epoch_seed(base_seed, i) for i in [0, 1, 2, 7]]
        samples = []
        for s in epoch_seeds:
            sampled = _sample_df(df, total_rows, random_state=s)
            samples.append(sampled)

        # Collect per-symbol min bar indices for each epoch
        min_bars_per_epoch = [_symbol_bar_min(s) for s in samples]

        # Assert epoch 0 differs from epoch 7 (acceptance criterion #2)
        e0 = min_bars_per_epoch[0]
        e7 = min_bars_per_epoch[3]
        for sym in e0:
            assert e0[sym] != e7[sym], (
                f"Symbol {sym}: epoch 0 min bar {e0[sym]} should differ "
                f"from epoch 7 min bar {e7[sym]}"
            )

        # Assert all epochs differ from each other
        for i in range(len(min_bars_per_epoch)):
            for j in range(i + 1, len(min_bars_per_epoch)):
                ei = min_bars_per_epoch[i]
                ej = min_bars_per_epoch[j]
                # At least one symbol has a different min bar
                diffs = [
                    sym for sym in ei if ei[sym] != ej[sym]
                ]
                assert diffs, (
                    f"Epoch {i} and epoch {j} have identical min bars "
                    f"for all symbols: {ei}"
                )

    def test_rotation_disabled_matches_legacy(self, monkeypatch):
        """PHASE2_PER_EPOCH_WINDOW_ROTATION=False → single start=0 behavior.

        When rotation is disabled, _resolve_sample_total_rows should not
        cap, and when n_per_sym > safe_len, _sample_df falls back to
        start=safe_start=0 (the legacy behavior).
        """
        monkeypatch.setattr(cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", False)
        df = _make_multi_sym_df(n_rows_per_sym=200, n_symbols=4)
        # n_per_sym = 701000/4 = 175250 > safe_len=200 → start = safe_start = 0
        sampled = _sample_df(df, 701_000)
        min_bars = _symbol_bar_min(sampled)
        for sym, bar_idx in min_bars.items():
            assert bar_idx == 0, (
                f"Symbol {sym}: expected min bar 0 (legacy fallback), "
                f"got {bar_idx}"
            )

    def test_warning_suppressed_when_capped(self, monkeypatch, caplog):
        """The 'exceeds largest safe range' warning does NOT fire when cap applied.

        With rotation enabled, _resolve_sample_total_rows caps total_rows so
        n_per_sym < safe_len, which means the RNG branch fires instead of
        the warning branch.
        """
        monkeypatch.setattr(cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True)
        df = _make_multi_sym_df(n_rows_per_sym=200, n_symbols=4)
        total_rows = _resolve_sample_total_rows(df, 701_000)
        # total_rows should be capped well below 701_000
        margin = max(1, 200 // 100)
        expected_max = 4 * (200 - margin)  # 4 * 198 = 792
        assert total_rows <= expected_max, (
            f"Capped total_rows {total_rows} exceeds expected max {expected_max}"
        )

        caplog.set_level(logging.WARNING)
        _sample_df(df, total_rows, random_state=42)

        # The specific warning about exceeding safe range should NOT appear
        warning_text = "exceeds largest safe range"
        assert warning_text not in caplog.text, (
            f"Warning '{warning_text}' was logged when it should have been "
            f"suppressed by the cap. caplog.text:\n{caplog.text}"
        )


class TestEndToEndRotation:
    """Integration-style tests with a mocked Rule_Pool_Generator."""

    @pytest.fixture
    def _patch_gen(self, monkeypatch):
        """Patch config for rotation and create a generator with minimal setup."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            Rule_Pool_Generator,
        )

        monkeypatch.setattr(cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True)
        monkeypatch.setattr(cfg, "PHASE2_USE_GPU", False)

        # Use 4 symbols × 2000 rows so the 1% margin gives ~21 possible
        # start positions → extremely unlikely that two different seeds
        # produce the same start.
        df = _make_multi_sym_df(n_rows_per_sym=2000, n_symbols=4)
        feature_infos = [
            {"name": "feat_0", "mode": "categorical"},
            {"name": "feat_1", "mode": "categorical"},
            {"name": "feat_2", "mode": "categorical"},
        ]

        gen = Rule_Pool_Generator(
            train_df=df,
            val_df=df.iloc[:100],
            feature_infos=feature_infos,
            direction="long",
            n_generations=10,
            seed=42,
            source_symbols=["SYM_0", "SYM_1", "SYM_2", "SYM_3"],
            island_id="0",
            island_profile="cluster",
            defer_warmup=True,
        )
        return gen

    def test_cached_scoped_train_kept(self, _patch_gen):
        """When rotation is enabled, _cached_scoped_train_df is stored."""
        gen = _patch_gen
        assert gen._cached_scoped_train_df is not None

    def test_resample_changes_cached_slim(self, _patch_gen):
        """After resample_train_for_epoch, the cached slim train changes."""
        gen = _patch_gen
        original_slim = gen._cached_slim_train.copy()

        gen.resample_train_for_epoch(1)
        new_slim = gen._cached_slim_train

        # The slim train should have different data after resampling
        # (different seed → different start bar with high probability)
        assert not original_slim.equals(new_slim), (
            "Resampled slim train is identical to original — "
            "this is extremely unlikely with different epoch seeds"
        )

    def test_resample_deterministic(self, _patch_gen):
        """Same epoch_idx produces identical cached slim train."""
        gen = _patch_gen
        gen.resample_train_for_epoch(3)
        slim_a = gen._cached_slim_train.copy()

        # Create another generator with same seed and resample the same epoch
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            Rule_Pool_Generator,
        )

        df = _make_multi_sym_df(n_rows_per_sym=2000, n_symbols=4, seed=42)
        feature_infos = [
            {"name": "feat_0", "mode": "categorical"},
            {"name": "feat_1", "mode": "categorical"},
            {"name": "feat_2", "mode": "categorical"},
        ]
        gen2 = Rule_Pool_Generator(
            train_df=df,
            val_df=df.iloc[:100],
            feature_infos=feature_infos,
            direction="long",
            n_generations=10,
            seed=42,
            source_symbols=["SYM_0", "SYM_1", "SYM_2", "SYM_3"],
            island_id="0",
            island_profile="cluster",
            defer_warmup=True,
        )
        gen2.resample_train_for_epoch(3)
        slim_b = gen2._cached_slim_train

        assert slim_a.equals(slim_b), (
            "Deterministic resample failed: same epoch seed produced "
            "different slim trains"
        )
