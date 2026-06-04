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
