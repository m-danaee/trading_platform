"""
detector.py — Feature_Detector

Classifies each feature column into exactly one of six modes:
  binary, ternary, positive, sparse_positive, sparse_signed, signed

The detection logic exactly mirrors evaluator_v5.ipynb's detect_feature_mode.
Mode detection MUST run on the training split only.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Feature_Detector:
    """Classify feature columns by their discretization type."""

    def detect_feature_mode(self, series: pd.Series) -> str:
        """
        Classify a single feature series into one of six modes.

        Detection order (matches evaluator_v5.ipynb exactly):
          1. binary        — unique non-NaN values ⊆ {0, 1}, count ≤ 2
          2. ternary       — unique non-NaN values ⊆ {-1, 0, 1}, count ≤ 3
          3. sparse_signed — min < 0 AND zero_ratio > 0.3
          4. signed        — min < 0 AND zero_ratio ≤ 0.3
          5. sparse_positive — min ≥ 0 AND zero_ratio > 0.3
          6. positive      — min ≥ 0 AND zero_ratio ≤ 0.3

        Notes:
          - zero_ratio is computed on the FULL series including zeros
            (not just non-NaN values), matching evaluator_v5.ipynb behaviour.
          - NaN values are excluded only for the unique-value checks in
            the binary/ternary branches.

        Parameters
        ----------
        series : pd.Series
            A single feature column from the training split.

        Returns
        -------
        str
            One of: "binary", "ternary", "positive", "sparse_positive",
            "sparse_signed", "signed".
        """
        unique_vals = series.dropna().unique()
        n_unique = len(unique_vals)

        if n_unique <= 2 and set(unique_vals).issubset({0, 1}):
            return "binary"
        if n_unique <= 3 and set(unique_vals).issubset({-1, 0, 1}):
            return "ternary"

        # zero_ratio on the full series (including zeros, excluding nothing)
        zero_ratio = (series == 0).mean()

        if series.min() < 0:
            return "sparse_signed" if zero_ratio > 0.3 else "signed"
        return "sparse_positive" if zero_ratio > 0.3 else "positive"

    def detect_all_modes(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict[str, str]:
        """
        Classify every column in *feature_cols* and return a mapping.

        Parameters
        ----------
        df : pd.DataFrame
            The training-split DataFrame (must NOT include validation/test rows).
        feature_cols : list[str]
            Column names to classify.  All must be present in *df*.

        Returns
        -------
        dict[str, str]
            ``{column_name: mode_string}`` for every column in *feature_cols*.
        """
        return {col: self.detect_feature_mode(df[col]) for col in feature_cols}


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

def detect_feature_mode(series: pd.Series) -> str:
    """Module-level convenience wrapper around Feature_Detector.detect_feature_mode."""
    return Feature_Detector().detect_feature_mode(series)


def detect_all_modes(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, str]:
    """Module-level convenience wrapper around Feature_Detector.detect_all_modes."""
    return Feature_Detector().detect_all_modes(df, feature_cols)
