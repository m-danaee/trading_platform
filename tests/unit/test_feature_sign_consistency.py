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


def test_check_spearman_sign_consistency_with_validation():
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
    
    # feature_c is consistent in train, but flips in val
    stable = _check_spearman_sign_consistency(train_df, ["feature_c"], n_folds=2, min_folds=1, val_df=val_df)
    assert "feature_c" not in stable

