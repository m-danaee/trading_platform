"""
symbol_conditions.py — Symbol filter parsing (evaluator_v5 parity).

Feature conditions are AND-ed. Symbol conditions within one rule are OR-ed,
then AND-ed with the feature mask. Rules without symbol filters apply to all
symbols.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

MAX_SYMBOL_FILTERS_PER_RULE = 10

_SYMBOL_CONDITION_RE = re.compile(
    r"^\s*\[?\s*symbol\s*\]?\s+is\s+(.+?)\s*$",
    flags=re.IGNORECASE,
)


def normalize_symbol_value(value: object) -> str:
    """
    Normalize symbol values so strategy conditions such as:
        "symbol is 1"
        "symbol is 1.0"
        "[symbol] IS 1"
    match dataset symbols consistently.
    """
    if pd.isna(value):
        return "__MISSING_SYMBOL__"

    text = str(value).strip()

    if (
        len(text) >= 2
        and ((text[0] == text[-1] == "'") or (text[0] == text[-1] == '"'))
    ):
        text = text[1:-1].strip()

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    if text.lower().startswith("symbol "):
        text = text.split(None, 1)[1].strip()

    try:
        numeric_value = float(text)
        if np.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(numeric_value))
    except (TypeError, ValueError):
        pass

    return text


def parse_symbol_condition(condition: object) -> list[str] | None:
    """
    Parse optional symbol filters.

    Supported formats:
        "symbol is 1"
        "symbol IS 1"
        "[symbol] IS 1"
        "symbol is 1,2,3"

    Returns a list of normalized symbol strings when the condition is a symbol
    filter, or None when it is a normal feature condition.
    """
    if not isinstance(condition, str):
        return None

    match = _SYMBOL_CONDITION_RE.match(condition)
    if match is None:
        return None

    value_text = match.group(1).strip()
    if not value_text:
        raise ValueError(f"Empty symbol value in condition: {condition!r}")

    raw_values = re.split(r"\s*,\s*", value_text)
    normalized_values: list[str] = []

    for raw_value in raw_values:
        normalized_value = normalize_symbol_value(raw_value)
        if not normalized_value:
            raise ValueError(f"Empty symbol value in condition: {condition!r}")
        normalized_values.append(normalized_value)

    return normalized_values


def split_feature_and_symbol_conditions(
    conditions: list[str],
    rule_number: int = 1,
) -> tuple[list[str], list[str]]:
    """
    Split normal feature conditions from optional symbol filters.

    Feature conditions are combined with AND.
    Symbol conditions inside one rule are combined with OR.
    If a rule has no symbol condition, it remains valid for all symbols.
    """
    if not isinstance(conditions, list) or len(conditions) == 0:
        raise ValueError(
            f"Rule {rule_number} must contain a non-empty 'conditions' list."
        )

    feature_conditions: list[str] = []
    symbol_values: list[str] = []

    for condition in conditions:
        parsed_symbols = parse_symbol_condition(condition)
        if parsed_symbols is None:
            feature_conditions.append(condition)
        else:
            symbol_values.extend(parsed_symbols)

    unique_symbols: list[str] = []
    seen_symbols: set[str] = set()

    for symbol_value in symbol_values:
        if symbol_value not in seen_symbols:
            unique_symbols.append(symbol_value)
            seen_symbols.add(symbol_value)

    if len(unique_symbols) > MAX_SYMBOL_FILTERS_PER_RULE:
        raise ValueError(
            f"Rule {rule_number} has {len(unique_symbols)} symbol filters. "
            f"A rule can include at most {MAX_SYMBOL_FILTERS_PER_RULE} symbol values."
        )

    return feature_conditions, unique_symbols


def get_normalized_symbol_array(df: pd.DataFrame) -> np.ndarray:
    """
    Build a normalized symbol array once per dataset.

    Uses factorization so normalization happens per unique symbol value instead
    of per row.
    """
    if "symbol" not in df.columns:
        raise ValueError(
            "This strategy contains a symbol filter, but the dataset has no "
            "'symbol' column."
        )

    codes, uniques = pd.factorize(df["symbol"], sort=False, use_na_sentinel=True)
    normalized_uniques = np.array(
        [normalize_symbol_value(value) for value in uniques],
        dtype=object,
    )

    normalized_symbols = np.empty(len(df), dtype=object)
    valid_mask = codes >= 0
    normalized_symbols[valid_mask] = normalized_uniques[codes[valid_mask]]
    normalized_symbols[~valid_mask] = "__MISSING_SYMBOL__"

    return normalized_symbols
