import numpy as np
import pandas as pd

from gpu_fuzzy_trader.features.selector import _check_spearman_sign_consistency


def test_check_spearman_sign_consistency():
    n = 300
    df = pd.DataFrame({
        "symbol": ["A"] * n,
        "feature_a": np.linspace(1, 10, n),
        "feature_b": np.hstack([np.linspace(1, 10, 100), np.linspace(10, 1, 100), np.linspace(1, 10, 100)]),
        "label_close_288": np.linspace(1, 10, n)
    })
    
    stable = _check_spearman_sign_consistency(df, ["feature_a", "feature_b"], n_folds=3, min_folds=2)
    assert "feature_a" in stable
    assert "feature_b" not in stable


def test_check_spearman_sign_consistency_ignores_validation():
    """Holdout validation is not used for sign-consistency (avoids val leakage)."""
    n = 100
    train_df = pd.DataFrame({
        "symbol": ["A"] * n,
        "feature_c": np.linspace(1, 10, n),
        "label_close_288": np.linspace(1, 10, n)
    })
    val_df = pd.DataFrame({
        "symbol": ["A"] * n,
        "feature_c": np.linspace(10, 1, n),
        "label_close_288": np.linspace(1, 10, n)
    })
    
    # feature_c is consistent in train folds; val flip must not affect selection
    stable = _check_spearman_sign_consistency(
        train_df, ["feature_c"], n_folds=2, min_folds=1, val_df=val_df, min_abs_corr=0.0,
    )
    assert "feature_c" in stable


def test_sign_consistency_ignores_noise_level_flip():
    """Opposite micro-signs across folds should pass when |rho| stays below threshold."""
    n = 400
    rng = np.random.default_rng(0)
    label = rng.normal(size=n)
    noise = rng.normal(scale=0.001, size=n)
    df = pd.DataFrame({
        "symbol": ["A"] * n,
        "feature_noise": label * 1e-4 + noise,
        "label_close_288": label,
    })

    stable = _check_spearman_sign_consistency(
        df, ["feature_noise"], n_folds=3, min_folds=2, min_abs_corr=0.02,
    )
    assert "feature_noise" in stable


def test_sign_consistency_still_drops_strong_flip():
    """Large opposite correlations across folds must still blacklist."""
    n = 300
    df = pd.DataFrame({
        "symbol": ["A"] * n,
        "feature_flip": np.hstack([
            np.linspace(1, 10, 100),
            np.linspace(10, 1, 100),
            np.linspace(1, 10, 100),
        ]),
        "label_close_288": np.linspace(1, 10, n),
    })

    stable = _check_spearman_sign_consistency(
        df, ["feature_flip"], n_folds=3, min_folds=2, min_abs_corr=0.02,
    )
    assert "feature_flip" not in stable
